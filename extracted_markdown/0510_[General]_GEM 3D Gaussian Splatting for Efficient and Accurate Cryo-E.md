<!-- page 1 -->
October 3, 2025
GEM: 3D Gaussian Splatting for Efficient and Accurate
Cryo-EM Reconstruction
Huaizhi Qu1, Xiao Wang2, Gengwei Zhang1, Jie Peng1, Tianlong Chen1,B
1University of North Carolina at Chapel Hill, 2University of Washington
{huaizhiq,tianlong}@cs.unc.edu, wang3702@uw.edu
Cryo-electron microscopy (cryo-EM) has become a central tool for high-resolution
structural biology, yet the massive scale of datasets (often exceeding 100k particle
images) renders 3D reconstruction both computationally expensive and memory
intensive. Traditional Fourier-space methods are efficient but lose fidelity due to
repeated transforms, while recent real-space approaches based on neural radiance
fields (NeRFs) improve accuracy but incur cubic memory and computation overhead.
Therefore, we introduce GEM, a novel cryo-EM reconstruction framework built on 3D
Gaussian Splatting (3DGS) that operates directly in real-space while maintaining high
efficiency. Instead of modeling the entire density volume, GEM represents proteins with
compact 3D Gaussians, each parameterized by only 11 values. To further improve the
training efficiency, we designed a novel gradient computation to 3D Gaussians that
contribute to each voxel. This design substantially reduced both memory footprint
and training cost. On standard cryo-EM benchmarks, GEM achieves up to 48× faster
training and 12× lower memory usage compared to state-of-the-art methods, while
improving local resolution by as much as 38.8%. These results establish GEM as a
practical and scalable paradigm for cryo-EM reconstruction, unifying speed, efficiency,
and high-resolution accuracy.
Project Page:
https://github.com/UNITES-Lab/GEM
1
Introduction
Cryo-electron microscopy (cryo-EM) (Bai et al., 2015a; Milne et al., 2013b) is a transformative biological
imaging technique that enables the determination of macromolecular structures at near-atomic resolution,
and its significance was recognized with the 2017 Nobel Prize in Chemistry (Pre). In cryo-EM, biological
samples are rapidly frozen in vitreous ice and imaged under an electron microscope, producing thousands
to millions of noisy two-dimensional projection images of individual particles. Each projection corresponds
to a different orientation of the same underlying macromolecule. By aggregating these projections,
the three-dimensional electron density of the protein can be reconstructed (Murata & Wolf, 2018).
The cryo-EM imaging model naturally aligns with the Fourier slice theorem, where each projection
corresponds to a central slice of the 3D Fourier transform of the density, enabling efficient reconstruction
in Fourier space (Zhong et al., 2021; Punjani et al., 2017; Scheres, 2012; Tang et al., 2007).
Despite its efficiency, the reliance on Fourier transforms introduces information loss in the traditional
methods, limiting achievable resolution. Recent works propose adapting neural radiance fields (NeRFs)
B Corresponding authors: tianlong@cs.unc.edu
Preprint. Under review.
arXiv:2509.25075v2  [cs.CV]  2 Oct 2025

<!-- page 2 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
to cryo-EM by modeling protein densities directly in real-space (Huang et al., 2024; Qu et al., 2025;
Liu et al., 2023), using representations such as MLPs (Liu et al., 2023), Transformers (Huang et al.,
2024), or multi-resolution hash encoding (Qu et al., 2025). These approaches remove the dependence on
Fourier transforms in the Fourier slice theorem (see Appendix A.1 for more details) and demonstrate
improved resolution compared to traditional Fourier-space pipelines. However, direct application of
NeRF to cryo-EM remains impractical. Unlike view synthesis tasks, cryo-EM requires reconstruction of
the full 3D density rather than rendering sparse pixels. And each training step involves applying the
contrast transfer function (CTF) across entire projection images, resulting in cubic scaling of memory
and computation. Even for small proteins, this requires millions of sampled points and gradients, making
NeRF-based cryo-EM reconstruction prohibitively slow and memory-intensive, especially on commodity
GPUs.
Memory
(GB ↓)
Speed
(it/s ↑)
Resolution
(Å ↓)
5.4
22.7
0.2
100.6
5.1
121.7
4.1
3.7
2.5
CryoDRGN
CryoNeRF
Ours
Figure 1: GEM achieves lower memory
usage, faster speed, and higher recon-
struction resolution compared to exist-
ing approaches.
To address these challenges, we propose GEM, a novel framework
that introduces 3D Gaussian Splatting (3DGS) (Kerbl et al.,
2023; Zha et al., 2024) for cryo-EM reconstruction. Unlike
NeRF-based approaches that model density implicitly, GEM
adopts an explicit representation using a set of 3D Gaussians,
each parameterized by only 11 values. This representation is
both compact and efficient: (i) 3D Gaussians are instantiated
only at nonzero-density regions, and during projection, (ii)
each 3D Gaussian contributes to multiple pixels and can be
rasterized in parallel. Furthermore, (iii) gradient computation
is restricted to the subset of Gaussians influencing a given
pixel, in contrast to NeRF methods that require activating the
entire network for each pixel. This design eliminates the cubic
memory footprint of NeRFs and yields substantial acceleration
in training.
We validate GEM on four widely used cryo-EM datasets un-
der diverse evaluation protocols. As shown in Figure 1, GEM
achieves substantially faster training and lower memory usage
than the real-space NeRF-based pipeline CryoNeRF, while
also surpassing both the Fourier-space baseline CryoDRGN and the real-space baseline CryoNeRF in
reconstruction quality. Across all datasets, GEM consistently attains gold-standard Fourier shell correlation
(GSFSC) resolutions better than 3 Å, outperforming existing methods. Additional evaluations further
confirm its resolution advantage, and compared to CryoNeRF (Qu et al., 2025), GEM not only improves
quality but also delivers significantly higher efficiency, even surpassing the Fourier-based CryoDRGN.
Our main contributions are summarized as follows:
• We introduce GEM, a novel framework that incorporates 3D Gaussian Splatting into cryo-EM
reconstruction, providing an explicit, efficient, and accurate representation of protein density that
avoids the information loss of Fourier-based methods and the cubic overhead of NeRF-based
real-space approaches.
• By explicitly modeling protein density with 3D Gaussians placed only in nonzero-density regions and
enabling parallel rasterization across pixels, GEM achieves up to 48× faster training speed compared
to existing methods.
• We design an efficient gradient computation strategy that restricts updates to only the 3D Gaussians
contributing to each pixel, thereby eliminating the cubic memory footprint and computation overhead
of prior real-space methods.
• Extensive experiments on four widely used cryo-EM datasets under diverse evaluation protocols
demonstrate that GEM consistently outperforms existing approaches, often reaching resolutions close
to the instrumental limitation.
2

<!-- page 3 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
2
Related Work
Cryo-EM reconstruction.
Classical single-particle cryo-EM pipelines formulate reconstruction in
Fourier space via the Fourier slice theorem, solving for the 3D density with iterative refinement (Zhong
et al., 2021; Punjani et al., 2017; Scheres, 2012; Tang et al., 2007). This formulation enables efficient
global solvers and robust regularization schemes and has underpinned much of the field’s progress (Milne
et al., 2013a; Murata & Wolf, 2018; Bai et al., 2015b). Quality is typically assessed using gold-standard
Fourier shell correlation (GSFSC) (Van Heel & Schatz, 2005), local resolution (Glaeser et al., 2021) and
directional consistency analyses (Huang et al., 2024). Despite their maturity, Fourier-space methods
involve repeated FFTs and interpolation steps and operate on band-limited representations, which can
incur information loss. Our work targets these pain points by adopting 3D Gaussian Splatting as an
explicit, efficient real-space representation while remaining compatible with standard cryo-EM pipelines
and evaluation protocols.
Real-space cryo-EM reconstruction.
A growing line of work reconstructs density directly in
Euclidean space using neural implicit fields and related parameterizations. Neural-field approaches,
which often use MLPs with Fourier or hash encodings, or Transformer-based fields, optimize a continuous
density so that its simulated projections match observed images, thereby avoiding frequency-domain
interpolation and enabling real-space regularization (Huang et al., 2024; Qu et al., 2025; Liu et al.,
2023; Herreros et al., 2025; Palmer & Aylett, 2022; Lu et al., 2022). These methods have demonstrated
improved resolution on several benchmarks. However, directly training neural fields for cryo-EM is
computationally demanding: unlike view synthesis, we must reconstruct full volumes rather than sparse
rays, apply CTFs per micrograph, and backpropagate through dense sampling, leading to memory/time
that scale roughly with the volume (and number of samples) instead of the number of visible pixels. This
cubic-like footprint makes large proteins and commodity GPUs challenging and motivates representations
whose gradients remain localized to the subset of primitives that actually contribute to each pixel.
3D Gaussian splatting.
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023; Zha et al., 2024; Wu et al.,
2024; Fei et al., 2024; Yu et al., 2024; Ye et al., 2025) represents scenes as explicit sets of anisotropic
Gaussians rendered by fast rasterization, yielding state-of-the-art real-time view synthesis with memory
that scales with the number of Gaussians rather than the discretized volume (Kerbl et al., 2023). For
biological settings, recent work proposes rectified radiative Gaussian splatting to improve fidelity under
line/slice integrals (Zha et al., 2024; 2022; Yu et al., 2025), suggesting that splatting is a promising
alternative to implicit fields when projections are the fundamental measurements. Adapting 3DGS
to cryo-EM introduces two key advantages: (i) sparsity, since Gaussians need only occupy non-empty
regions of the macromolecule, and (ii) locality, since gradients for a pixel involve only the Gaussians
that cover it, avoiding the cubic memory of dense neural fields. Our framework instantiates these ideas
for single-particle cryo-EM by coupling a CTF-aware, differentiable projection of Gaussians with joint
optimization over density and imaging latents.
3
Method
3.1
Cryo-EM Reconstruction
Cryo-EM reconstruction (Elmlund & Elmlund, 2015) aims to recover the 3D structure of a protein from
a large collection of noisy 2D particle images. In practice, each image is acquired as a projection of
a protein particle illuminated by an electron beam, modulated by the microscope’s contrast transfer
function (CTF) and corrupted by background noise.
Image Formation Model.
Let I = {Ii}N
i=1 denote a dataset of N particle images and V = {Vi}N
i=1
the corresponding particles. Each image Ii ∈Rd×d in Figure 3 is a projection of an underlying 3D
density map Vi ∈Rd×d×d following the imaging model that can be expressed as
Ii = Ci ∗Proj (Vi; ϕi, ti) + ηi,
(3.1)
where Ci is the CTF determined by microscope settings and ∗is the convolution operation. Proj(V ) =
R 0
−∞V dz denotes the projection operator with rotation angles ϕi and translation ti, and ηi represents
additive noise.
3

<!-- page 4 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Electron Beam
Figure 2:
Cryo-EM imaging process.
Randomly oriented proteins are illumi-
nated by the electron beam, leaving
traces on the sensor as particle images
I1, I2, and I3. In this work we focus on
homogeneous reconstruction, where all
particles V1, V2, and V3 share the same
underlying 3D structure and differ only
in orientation (i.e., multiple copies of
the same object).
Homogeneous Reconstruction.
In this work, we focus
on homogeneous reconstruction, where all particles, e.g. V1,
V2, and V3 in Figure 3, share the same underlying structure:
Vi = V, ∀Vi ∈V.
(3.2)
The goal is then to recover the shared 3D density V from all
images and their associated parameters:
bV = Reconstruct (I; Φ, T , C) ,
(3.3)
where Φ = {ϕi}N
i=1, T = {ti}N
i=1, and C = {Ci}N
i=1 denote
the set of rotations, translations and CTFs, which are known
parameters before the reconstruction.
Fourier Slice Theorem for Reconstruction: Fourier-
space methods leverage the Fourier slice theorem and there-
fore only need to predict a slice in the Fourier domain, i.e.,
Slice(F(bVi); , ϕi, ti), without instantiating the full protein
density bV . Real-Space Cryo-EM Reconstruction: In
contrast, real-space methods instantiate the entire volume
bV and perform projection following Equation 3.1, which re-
quires cubic memory to store both the predicted values and
their gradients.
3.2
3D Gaussian Splatting for Reconstruction
We present the GEM training framework in Figure 4. First,
protein density can be represented as a set of 3D Gaussians (Figure 3a). To optimize the randomly
initialized 3D Gaussian-based representation, we can project 3D Gaussian representations to 2D space.
The 3D Gaussian representations can then be optimized by minimizing the differences between the
projected 2D images and the captured 2D images by the real microscope. During training, we also
introduced novel efficient gradient computation to reduce memory cost. After training, the reconstructed
protein density can be queried from the trained 3D Gaussians (Figure 3b).
Protein Density represented by 3D Gaussians.
Following previous 3DGS literature (Kerbl et al.,
2023; Zha et al., 2024), we model the protein density as a set of 3D Gaussians G = {Gj}M
j=1, where each
Gaussian is parameterized by its center pj, covariance matrix Σj, and density coefficient ρj. Each kernel
Gj defines a local Gaussian-shaped density field:
Gj(x | ρj, pj, Σj) = ρj · exp

−1
2(x −pj)⊤Σ−1
j (x −pj)

,
Σj = RjSjS⊤
j R⊤
j .
(3.4)
where the covariance Σj is further decomposed into a rotation Rj and a scale matrix Sj. The overall
density of protein Vi at position x is then given by the summation of all Gaussian kernels:
bVi(x) =
M
X
j=1
Gj(x | ρj, pj, Σj).
(3.5)
Compared to the original 3DGS formulation, which incorporates view-dependent effects for neural
rendering, our variant adopts an isotropic, density-based, density-based formulation consistent with the
cryo-EM imaging model, where reconstruction relies on where reconstruction relies on direct density
integration along the electron beam.
Fast 2D Projection of 3D Gaussian representations via Explicit Integration for .
Owing
to the property of 3D Gaussians, the projection can be expressed in a closed form, avoiding querying
density values and then aggregating them. Let bVi(x) denote the protein density for view i. The volume
4

<!-- page 5 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
is defined as a sum of 3D 3D Gaussians following Equation 3.5, and the integration along the optical
axis z yields a 2D Gaussian mixture:
bIi(r) =
M
X
j=1
ρj
1
2π
bΣj
1/2 exp

−1
2 (bx −bpj)⊤bΣ−1
j (bx −bpj)

=
M
X
j=1
G2
j,
(3.6)
where bx ∈R2, bpj ∈R2, and bΣj ∈R2×2 are the in-plane (mean, covariance) obtained by dropping the
third row/column of their 3D counterparts x, pj, and Σj, respectively. This closed-form marginalization
enables efficient CUDA kernels and removes the need to explicitly query bVi on 3D grids before the
projection, substantially reducing both compute and memory. Please see Appendix A.2 for more detail
of the derivation of the projection. As shown in Figure 4, Step1 applies Equation 3.6 to generate the
projection from the protein density.
Optimizing 3D GS representations
Given the projected image bIi, we apply the CTF Ci in Fourier
space and compare to the noisy measurement Ii via an ℓ2 loss:
Ipred
i
= F−1 F(Ci) · F(bIi)

,
Li =
Ipred
i
−Ii
2
2,
(3.7)
where F and F−1 are Fourier and inverse Fourier transforms, respectively. In Equation 3.7, Fourier
transform is utilized as the convolution in Equation 3.1 equals to the multiplication in Fourier space
(McGillem & Cooper, 1991). Although 3DGS is typically trained on clean natural images, we observe
that the standard MSE loss is sufficient for noisy cryo-EM images. Equation 3.7 corresponds to Step 2
and 3 in Figure 4, where the CTF is first applied to the projected image, followed by computing the
standard MSE loss between the CTF-corrected image and the experimental ground-truth image.
Efficient Gradient Computation to Reduce Memory Overhead.
While our rasterization
implementation follows the principle of Equation 3.6, directly evaluating all 3D Gaussians remains
inefficient. To address this, we introduce a thresholding strategy that selects only the Gaussians with
non-negligible contributions to a given ray r. Formally,
bIi(r) =
K
X
j=1
G2
j,
where G2
j > τ,
(3.8)
where τ is a predefined threshold. This pruning avoids unnecessary computation from Gaussians far
from the ray, since their contribution decays quadratically with distance to the Gaussian center pj.
After selection, the remaining 3D Gaussians are sorted along the z-axis, and the accumulation in
Equation 3.8 is performed from the lowest z-value (closest to the cryo-EM sensor) to the highest (farthest
from the sensor). This ordering prioritizes the most influential 3D Gaussians during rendering and
further improves computational efficiency.
4
Experiment
In this section, we evaluate the proposed method GEM using four widely adopted cryo-EM datasets:
EMPIAR-10005 (TRPV1, Liao et al. (2013)), EMPIAR-10028 (Plasmodium falciparum 80S ribosome,
Wong et al. (2014)), EMPIAR-10049 (Synaptic RAG1-RAG2 complex, Ru et al. (2015)), and EMPIAR-
10076 (L17-depleted 50S ribosomal intermediates, Davis et al. (2016)).
To demonstrate the effectiveness of GEM, we compare it with two representative cryo-EM reconstruction
approaches: (i) the conventional voxel-based method CryoSPARC (Punjani et al., 2017), and (ii) the
neural network-based CryoDRGN (Zhong et al., 2021). Both of the methods reconstruct cryo-EM
densities in Fourier space levaraging the Fourier slice theorem (Radon, 2005; Garces et al., 2011), in
contrast to our GEM that operates in 3D Euclidean space. For CryoSPARC, we first perform ab initio
reconstruction, followed by homogeneous refinement. For CryoDRGN, we employ the train_nn function,
which is designed for homogeneous reconstruction without modeling heterogeneity. In both cases, we use
their default parameters. In the following sections, we first evaluate the reconstruction efficiency, and
5

<!-- page 6 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Protein Density
Initialized 3DGS
Initialize
(a) Protein density is represented by 3D Gaussians.
Before training, the parameters of 3D Gaussians
are randomly initialized.
(Eq. 5)
Reconstructed
Density
Density
Query
Trained 3DGS
(b) After training, the density reconstruction is
queried from the trained 3D Gaussians following
Equation 3.5.
Figure 3: Initialization before training and density reconstruction after training.
Step 1:
Projection
(Eq. 8)
Step 2:
Apply CTF
(Eq. 10)
Step 3:
Compute Loss
(Eq. 10)
Projected Image
Corrupted Image
GT Image
Trained
Initialized 3DGS
Trained 3DGS
Figure 4: The overview of GEM training. The training begins with randomly initialized Gaussians. The
3D Gaussians are projected following Equation 3.6. Then the CTF is applied (Equation 3.7) to the
projection and being compared with the noisy experimental image to calculate the loss (Equation 3.7).
Table 1: Efficiency comparison of deep learning-based cryo-EM reconstruction methods. Speed (it/s)
measures the number of images processed per second, an Memor (GB) reports the maximum GPU
memory consumption during training. All experiments are conducted on a single NVIDIA RTX A6000 (48
GB). OOM indicates out-of-memory errors. Our method achieves the highest training throughput
while requiring substantially less memory across all datasets.
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
Speed (it/s) ↑
Memory (GB) ↓
Speed (it/s) ↑
Memory (GB) ↓
Speed (it/s) ↑
Memory (GB) ↓
Speed (it/s) ↑
Memory (GB) ↓
CryoDRGN
47.62
8.81
142.86
2.58
58.82
8.79
100.00
4.93
CryoNeRF
-
OOM
9.13
9.94
2.10
37.75
4.03
20.50
Ours
94.10
1.54
160.49
0.50
96.60
0.74
108.11
0.51
then examnine the reconstructed protein densities from GEM, CryoSPARC and CryoDRGN with three
mainstream evaluation protocols: gold-standard Fourier shell correlation (Harauz & van Heel (1986),
Section 4.2), local resolution estimation (Adams et al. (2010), Section 4.3), and Fourier slice correlation
(Huang et al. (2024), Section 4.4).
4.1
Efficiency Comparison
We first evaluate the efficiency of GEM against deep learning-based cryo-EM reconstruction methods.
Specifically, we compare with CryoDRGN (Fourier-based) and CryoNeRF (NeRF-based). All experiments
are conducted on a single NVIDIA RTX 6000 GPU, and we report both training throughput (images
per second) and peak GPU memory usage. As shown in Table 1, GEM consistently achieves the best
efficiency, with substantially lower memory usage and faster speed than the baselines. CryoNeRF suffers
from cubic memory and computation overhead inherent to NeRF, resulting in very slow training and
frequent out-of-memory (OOM) failures. CryoDRGN, which leverages the Fourier slice theorem with
quadratic complexity, is faster than CryoNeRF but still significantly outperformed by GEM. Moreover,
GEM exhibits more stable memory usage, as it depends primarily on the number of 3D Gaussians used to
represent the protein density.
6

<!-- page 7 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Figure 5: GSFSC of our GEM and baselines. The horizontal axis denotes resolution in ångströms (Å),
and the vertical axis denotes the FSC value. The two dashed horizontal lines are FSC thresholds of 0.5
and 0.143. The final resolution is defined as the resolution at which the GSFSC curve first drops below
the 0.143 threshold (smallest possible resolution if no intersection). Intersection points further to
the right corresponds to better reconstruction quality.
Table 2: Final resolutions (in Å) obtained by each method according to the GSFSC 0.143 threshold.
Lower values indicate better resolution.
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
CryoSPARC
4.12 Å
4.53 Å
3.39 Å
4.50 Å
CryoDRGN
3.81 Å
4.02 Å
3.66 Å
3.27 Å
Ours
2.98 Å
2.56 Å
2.62 Å
2.43 Å
Table 3: Mean local resolution (in Å) of reconstructions on the four cryo-EM datasets. Lower values
correspond to higher resolution.
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
CryoSPARC
3.55 Å
3.57 Å
3.16 Å
5.54 Å
CryoDRGN
3.39 Å
3.22 Å
3.22 Å
3.38 Å
Ours
2.87 Å
2.62 Å
2.99 Å
2.58 Å
4.2
Gold-Standard Fourier Shell Correlation
Gold-standard Fourier shell correlation (GSFSC) (Van Heel & Schatz, 2005) is a commonly used
evaluation to quantify the reconstruction quality of a cryo-EM reconstruction method on a dataset.
Given a cryo-EM particle image dataset I = {Ii}N
i=1, to evaluate a method with GSFSC, the dataset I
is first randomly splitted into two non-overlapping splits I1 and I2. Then the method to test is applied
on the two splits to produce two reconstructions X1 and X2. The FSC value at spatial frequency shell
ri in the Fourier space is defined as:
FSC(ri) =
P
r∈ri F1(r) · F2(r)∗
qP
r∈ri |F1(r)|2 · P
r∈ri |F2(r)|2 ,
(4.1)
where F1(r) and F2(r) denote the Fourier coefficients of X1 and X2 at frequency r, and ∗indicates
complex conjugation. The summations are computed over all voxels in the shell ri. As in the GSFSC
protocol the particle images are split into two halves I1 and I2 and reconstructed independently, it
ensures that the correlation reflects reproducible structural information rather than overfitting.
Since the spatial frequency r is inversely related to the 3D Euclidean space resolution, the GSFSC
values can be plotted as a curve of FSC values versus resolutions (in Å). The resolution estimate is then
determined by the point at which the GSFSC curve drops below a predefined threshold (commonly
0.143). This crossing point represents the highest spatial frequency (lowest resolution value) at which
7

<!-- page 8 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
EMPIAR-10076
EMPIAR-10049
CryoSPARC
EMPIAR-10028
CryoDRGN
Ours
EMPIAR-10005
Figure 6: Local resolution maps of reconstructions from GEM, CryoSPARC, and CryoDRGN. Each map
is colored according to the estimated local resolution, with the color scale ranging from 2 Å (blue, higher
resolution) to 5 Å (red, lower resolution). Better reconstructions are indicated by larger regions
in blue-green.
Table 4: FSLC results on four cryo-EM datasets. Each row reports the mean FSLC value with its
standard deviation across randomly sampled orientations. Higher mean values indicate stronger
directional consistency, and lower standard deviation reflects greater stability.
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
CryoSPARC
0.961 ± 0.030
0.348 ± 0.160
0.978 ± 0.012
0.886 ± 0.084
CryoDRGN
0.989 ± 0.005
0.814 ± 0.109
0.976 ± 0.020
0.974 ± 0.025
Ours
0.998 ± 0.002
0.971 ± 0.022
0.992 ± 0.010
0.982 ± 0.019
the reconstruction remains reliable, thereby providing a quantitative resolution assessment.
As shown in Figure 5 and Table 2, ❶GEM consistently achieves higher resolutions (i.e., lower Å values at
the intersection with the 0.143 FSC threshold) than the baselines across all datasets. It is noteworthy
that, according to the Nyquist-Shannon sampling theorem (Shannon, 2006), the ultimate attainable
resolution is limited to twice the voxel size of the protein density map, corresponding to 2.46 Å and
2.62 Å for EMPIAR-10049 and EMPIAR-10028, respectively. ❷On these two datasets, GEM achieves
resolutions that approach this theoretical upper bound, highlighting its strong ability to recover fine
structural details from cryo-EM images.
4.3
Local Resolution Estimation
We then evaluate the reconstructions from all methods using the standard protocol of local resolution
estimation. While the FSC provides a single global resolution value, the same principle can be localized to
8

<!-- page 9 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
Figure 7: FSLC maps of reconstructions from GEM, CryoSPARC, and CryoDRGN across four datasets.
Each map depicts the correlation values of slices sampled across different elevation and azimuth angles,
with the color scale ranging from low correlation (purple) to high correlation (yellow). More uniform
yellow regions indicate higher and more isotropic directional resolution.
regions within the protein density, providing more detailed estimation of the quality of the reconstruction.
Specifically, given two half maps reconstructed from independent data splits, a cubic subregion is
extracted at the same location from each map. The FSC between the two cubes is then computed
following Equation 4.1, and the local resolution is determined from the frequency at which the FSC
curve falls below the 0.143 threshold. This estimated resolution is assigned to the voxel at the cube
center. Repeating this process across the entire map yields a resolution map that depicts the spatial
variability of the reconstruction quality.
As shown in Figure 6 and Table 3, GEM consistently achieves superior local resolution across all four
datasets. The EMPIAR-10005 dataset poses a significant challenge to existing approaches due to its
high symmetry and structural complexity, leading CryoSPARC and CryoDRGN to achieve average
local resolutions of approximately 5 Å and 3 Å, respectively. In contrast, GEM attains an average local
resolution of 2.28 Å, which approaches the Nyquist-Shannon limit of 2.43 Å, thereby demonstrating its
strong capability to recover fine structural details.
4.4
Fourier Slice Correlation
We further validate the reconstructions using a recently proposed evaluation protocol, Fourier slice
correlation (FSLC, Huang et al. (2024)). FSLC serves as an intermediate-level metric that quantifies
directional resolution anisotropy by measuring the similarity between corresponding slices extracted
from two reconstructed protein densities (Tan et al., 2017; Huang et al., 2024). In practice, slices are
sampled at uniformly distributed elevation angles (0° −180°) and azimuth angles (0° −360°).
9

<!-- page 10 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Table 5: Ablation study on different parameters of 3DGS. "No Rotation" means the rotation of all 3D
Gaussians are fixed at 0. "Isotropic Scale" indicates the scale matrix Si is identical across the three axis
of the 3D Gaussians, resulting in a spherical shape of the 3D Gaussians.
EMPIAR-10028
EMPIAR-10049
EMPIAR-10076
EMPIAR-10005
❶No Rotation
3.71 Å
4.23 Å
5.99 Å
4.15 Å
❷Isotropic Scaling
4.02 Å
5.53 Å
12.80 Å
4.21 Å
❶+ ❷
10.67 Å
6.81 Å
13.54 Å
7.95 Å
Original
2.98 Å
2.56 Å
2.62 Å
2.43 Å
As shown in Figure 7 and Table 4, GEM consistently achieves higher FSLC values, indicating stronger
agreement across a wide range of rotation angles. While all methods exhibit relatively lower correlation
around the azimuthal regions of the elevation axis, GEM still outperforms the baselines, demonstrating
superior robustness in challenging orientations. Moreover, our method attains the lowest standard
deviation across the four datasets, further highlighting its stability and consistency with respect to
directional variations.
4.5
Design Choices of 3DGS for cryo-EM
While 3DGS was developed for reconstruction of natural scenes, it is unclear which set of parameters
are essential for cryo-EM reconstruction. We therefore ablate per-Gaussian rotation and anisotropic
scaling. Concretely, we evaluate: (i) No Rotation, fix Ri = I; (ii) Isotropic Scaling, enforce Si = siI,
where si is the distance of the i-th 3D Gaussian to its nearest neighbor; (iii) No Rotation and Isotropic
Scaling (both constraints).
As shown in Table 5 (lower Å is better), ❶all ablations degrade resolution across datasets. ❷Enforcing
isotropic scaling is consistently more harmful than removing rotation (e.g., on EMPIAR-10076, 2.62 Å to
12.80 Å vs. 2.62 Å to 5.99 Å), and combining both constraints cause the most harm to the performance.
These results indicate that both per-Gaussian rotation and anisotropic scaling are critical for accurately
modeling cryo-EM densities, validating our choice to retain the full 3DGS parameterization in GEM.
5
Conclusion
In this work, we introduced GEM, a novel framework that leverages 3D Gaussian Splatting (3DGS) for
cryo-EM reconstruction. By explicitly representing protein densities with 3D Gaussians, GEM avoids the
information loss of Fourier-space methods and the cubic memory and computation overhead of NeRF-
based real-space approaches, thereby achieving high resolution and efficiency. Through comprehensive
experiments, GEM shows that it delivers up to 48× faster training, 12× lower memory consumption, and
up to 38.8% improvement in local resolution over prior state-of-the-art methods, often approaching the
physical resolution limit of the microscope. Overall, GEM establishes 3DGS as a practical and scalable
paradigm for cryo-EM, unifying efficiency with high-resolution accuracy and offering broad benefits to
the structural biology community.
Acknowledgement
This research was partially funded by the National Institutes of Health (NIH) under award 1R01EB037101-
01. The views and conclusions contained in this document are those of the authors and should not be
interpreted as representing the official policies, either expressed or implied, of the NIH. Tianlong Chen
was also partially supported by the Amazon Research Award (Spring 2025) and the Gemma Academic
Program GCP Credit Award.
10

<!-- page 11 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
References
Press release: The 2017 Nobel Prize in Chemistry. 1
Adams, P. D., Afonine, P. V., Bunkóczi, G., Chen, V. B., Davis, I. W., Echols, N., Headd, J. J., Hung,
L.-W., Kapral, G. J., Grosse-Kunstleve, R. W., et al. Phenix: a comprehensive python-based system
for macromolecular structure solution. Biological crystallography, 66(2):213–221, 2010. 6
Bai, X.-C., McMullan, G., and Scheres, S. H. How cryo-em is revolutionizing structural biology. Trends
in biochemical sciences, 40(1):49–57, 2015a. 1
Bai, Y., Müller, D. B., Srinivas, G., Garrido-Oter, R., Potthoff, E., Rott, M., Dombrowski, N., Münch,
P. C., Spaepen, S., Remus-Emsermann, M., et al. Functional overlap of the arabidopsis leaf and root
microbiota. Nature, 528(7582):364–369, 2015b. 3
Bracewell, R. N. Strip integration in radio astronomy. Australian Journal of Physics, 9(2):198–217, 1956.
14
Davis, J. H., Tan, Y. Z., Carragher, B., Potter, C. S., Lyumkis, D., and Williamson, J. R. Modular
assembly of the bacterial large ribosomal subunit. Cell, 167(6):1610–1622, 2016. 5
Elmlund, D. and Elmlund, H. Cryogenic electron microscopy and single-particle analysis. Annual review
of biochemistry, 84(1):499–517, 2015. 3
Fei, B., Xu, J., Zhang, R., Zhou, Q., Yang, W., and He, Y. 3d gaussian splatting as new era: A survey.
IEEE Transactions on Visualization and Computer Graphics, 2024. 3
Garces, D. H., Rhodes, W. T., and Peña, N. M. Projection-slice theorem: a compact notation. Journal
of the Optical Society of America A, 28(5):766–769, 2011. 5
Glaeser, R. M., Nogales, E., and Chiu, W. Single-particle Cryo-EM of biological macromolecules. IOP
publishing, 2021. 3
Harauz, G. and van Heel, M. Exact filters for general geometry three dimensional reconstruction. Optik.,
73(4):146–156, 1986. 6
Herreros, D., Mata, C. P., Noddings, C., Irene, D., Krieger, J., Agard, D. A., Tsai, M.-D., Sorzano, C.
O. S., and Carazo, J. M. Real-space heterogeneous reconstruction, refinement, and disentanglement of
cryoem conformational states with hetsiren. Nature communications, 16(1):3751, 2025. 3
Huang, Y., Zhu, C., Yang, X., and Liu, M. High-resolution real-space reconstruction of cryo-em structures
using a neural field network. Nature Machine Intelligence, 6(8):892–903, 2024. 2, 3, 6, 9, 14
Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. 3d gaussian splatting for real-time radiance
field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 2, 3, 4
Liao, M., Cao, E., Julius, D., and Cheng, Y. Structure of the trpv1 ion channel determined by electron
cryo-microscopy. Nature, 504(7478):107–112, 2013. 5
Liu, X., Zeng, Y., Qin, Y., Li, H., Zhang, J., Xu, L., and Yu, J.
Cryoformer: Continuous het-
erogeneous cryo-em reconstruction using transformer-based neural representations. arXiv preprint
arXiv:2303.16254, 2023. 2, 3
Lu, Y., Liu, J., Zhu, L., Zhang, B., and He, J. 3d reconstruction from cryo-em projection images using
two spherical embeddings. Communications Biology, 5(1):304, 2022. 3
McGillem, C. D. and Cooper, G. R. Continuous and discrete signal and system analysis. (No Title),
1991. 5
Milne, I., Stephen, G., Bayer, M., Cock, P. J., Pritchard, L., Cardle, L., Shaw, P. D., and Marshall, D.
Using tablet for visual exploration of second-generation sequencing data. Briefings in bioinformatics,
14(2):193–202, 2013a. 3
11

<!-- page 12 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Milne, J. L., Borgnia, M. J., Bartesaghi, A., Tran, E. E., Earl, L. A., Schauder, D. M., Lengyel,
J., Pierson, J., Patwardhan, A., and Subramaniam, S. Cryo-electron microscopy–a primer for the
non-microscopist. The FEBS journal, 280(1):28–45, 2013b. 1
Murata, K. and Wolf, M.
Cryo-electron microscopy for structural analysis of dynamic biological
macromolecules. Biochimica et Biophysica Acta (BBA)-General Subjects, 1862(2):324–334, 2018. 1, 3
Palmer, C. M. and Aylett, C. H. Real space in cryo-em: the future is local. Biological Crystallography,
78(2):136–143, 2022. 3
Punjani, A., Rubinstein, J. L., Fleet, D. J., and Brubaker, M. A. cryosparc: algorithms for rapid
unsupervised cryo-em structure determination. Nature methods, 14(3):290–296, 2017. 1, 3, 5, 14
Qu, H., Wang, X., Zhang, Y., Wang, S., Noble, W. S., and Chen, T. Cryonerf: reconstruction of
homogeneous and heterogeneous cryo-em structures using neural radiance field. bioRxiv, pp. 2025–01,
2025. 2, 3, 14
Radon, J. 1.1 über die bestimmung von funktionen durch ihre integralwerte längs gewisser mannig-
faltigkeiten. Classic papers in modern diagnostic radiology, 5(21):124, 2005. 5
Ru, H., Chambers, M. G., Fu, T.-M., Tong, A. B., Liao, M., and Wu, H. Molecular mechanism of v (d)
j recombination from synaptic rag1-rag2 complex structures. Cell, 163(5):1138–1152, 2015. 5
Scheres, S. H. Relion: implementation of a bayesian approach to cryo-em structure determination.
Journal of structural biology, 180(3):519–530, 2012. 1, 3
Shannon, C. E. Communication in the presence of noise. Proceedings of the IRE, 37(1):10–21, 2006. 8
Tan, Y. Z., Baldwin, P. R., Davis, J. H., Williamson, J. R., Potter, C. S., Carragher, B., and Lyumkis, D.
Addressing preferred specimen orientation in single-particle cryo-em through tilting. Nature methods,
14(8):793–796, 2017. 9
Tang, G., Peng, L., Baldwin, P. R., Mann, D. S., Jiang, W., Rees, I., and Ludtke, S. J. Eman2: an
extensible image processing suite for electron microscopy. Journal of structural biology, 157(1):38–46,
2007. 1, 3
Van Heel, M. and Schatz, M. Fourier shell correlation threshold criteria. Journal of structural biology,
151(3):250–262, 2005. 3, 7
Wong, W., Bai, X.-c., Brown, A., Fernandez, I. S., Hanssen, E., Condron, M., Tan, Y. H., Baum, J.,
and Scheres, S. H. Cryo-em structure of the plasmodium falciparum 80s ribosome bound to the
anti-protozoan drug emetine. elife, 3:e03080, 2014. 5
Wu, T., Yuan, Y.-J., Zhang, L.-X., Yang, J., Cao, Y.-P., Yan, L.-Q., and Gao, L. Recent advances in 3d
gaussian splatting. Computational Visual Media, 10(4):613–642, 2024. 3
Ye, V., Li, R., Kerr, J., Turkulainen, M., Yi, B., Pan, Z., Seiskari, O., Ye, J., Hu, J., Tancik, M., et al.
gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):
1–17, 2025. 3
Yu, W., Cai, Y., Zha, R., Fan, Z., Li, C., and Yuan, Y. X2-gaussian: 4d radiative gaussian splatting for
continuous-time tomographic reconstruction. arXiv preprint arXiv:2503.21779, 2025. 3
Yu, Z., Chen, A., Huang, B., Sattler, T., and Geiger, A. Mip-splatting: Alias-free 3d gaussian splatting. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19447–19456,
2024. 3
Zha, R., Zhang, Y., and Li, H. Naf: neural attenuation fields for sparse-view cbct reconstruction.
In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp.
442–452. Springer, 2022. 3
12

<!-- page 13 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Zha, R., Lin, T. J., Cai, Y., Cao, J., Zhang, Y., and Li, H. R2-gaussian: Rectifying radiative gaussian
splatting for tomographic reconstruction. arXiv preprint arXiv:2405.20693, 2024. 2, 3, 4
Zhong, E. D., Bepler, T., Berger, B., and Davis, J. H. Cryodrgn: reconstruction of heterogeneous
cryo-em structures using neural networks. Nature methods, 18(2):176–185, 2021. 1, 3, 5, 14
Zwicker, M., Pfister, H., Van Baar, J., and Gross, M. Ewa volume splatting. In Proceedings Visualization,
2001. VIS’01., pp. 29–538. IEEE, 2001. 15
Zwicker, M., Pfister, H., Van Baar, J., and Gross, M. Ewa splatting. IEEE Transactions on Visualization
and Computer Graphics, 8(3):223–238, 2002. 15
13

<!-- page 14 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Appendix
A Appendix
14
A.1 Description of Baseline Methods
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
A.2 Derivation of 3DGS Projection
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
A.3 Description of Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
A
Appendix
A.1
Description of Baseline Methods
Fourier Slice Theorem for Reconstruction.
A common strategy in cryo-EM reconstruction (Zhong
et al., 2021; Punjani et al., 2017) is to leverage the Fourier slice theorem (Bracewell, 1956), which states
that the Fourier transform of a 2D projection of a 3D density Vi corresponds to a central slice of its 3D
Fourier transform perpendicular to the projection direction. Formally,
F(Proj(Vi; ϕi, ti)) = Slice(F(Vi); ϕi, ti) ,
(A.1)
where F(·) denotes the Fourier transform and Slice(·; ϕi, ti) extracts the 2D slice orthogonal to the
projection direction specified by (ϕi, ti). This theorem implies that reconstructing the protein density
reduces to learning to predict individual slices in Fourier space, thereby significantly lowering the memory
footprint during training.
Real-Space Cryo-EM Reconstruction.
The reconstruction formulation in Equation 3.3 closely
parallels 3D reconstruction problems in computer vision, which has motivated recent work to perform
cryo-EM reconstruction directly in real space using neural radiance fields (NeRFs) (Huang et al., 2024;
Qu et al., 2025). Unlike Fourier-based methods, this approach avoids repeated Fourier transforms and
can in principle achieve higher resolution. However, it requires predicting the full density bVi before
supervision can be applied, since the contrast transfer function (CTF) Ci must operate on the complete
projection image due to the convolution operation.
Concretely, given a set of predefined coordinates D, each particle’s pose is applied as
Di = Rotate(D; ϕi) + ti,
(A.2)
and the resulting coordinates are used to query the NeRF, producing the predicted density volume bVi.
The model is then trained by projecting bVi under the same pose and computing the loss between the
simulated projection bIi and the experimental image Ii:
Li = MSE

Ii, Proj

bVi; ϕi, ti

.
(A.3)
Because the entire density bVi must be generated at each iteration, this method incurs cubic memory and
computation cost, making training slow and resource-intensive despite its resolution advantage.
A.2
Derivation of 3DGS Projection
The j-th 3D Gaussian is defined as:
Gj(x | ρj, pj, Σj) = ρj · exp

−1
2(x −pj)⊤Σ−1
j (x −pj)

.
(A.4)
14

<!-- page 15 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
Therefore for a query position x in the protein density, its value is defined as
bVi(x) =
M
X
j=1
Gj(x | ρj, pj, Σj).
(A.5)
To avoid the query of the entire protein density before the projection, we aim to explicitly derive the
expression of the projected image bIi. Since each pixel in the image bIi corresponds to an electron beam r,
we denote the projection of this specific pixel as bIi(r).
Following the definition, the projection of the pixel can be defined as:
bIi(r) =
Z
bVi(x) dz =
Z
M
X
j=1
Gj(x | ρj, pj, Σj) dz
(A.6)
=
M
X
j=1
Z
Gj(x | ρj, pj, Σj) dz.
(A.7)
From the standard 3DGS for natural images, we have
bIr(r) ≈
M
X
j=1
Z
Gj(x|ρj, pj, JjWΣjW⊤J⊤
j
|
{z
}
eΣi
) dz,
(A.8)
where eΣi is the new Gaussian covariance matrix controlled by local approximation matrix Jj and
viewing transformation matrix W that only corresponds to the rotation ϕ. The projection of natural
images follows the pinhole camera model, while in cryo-EM the projection rays are parallel, therefore
the Jacobian matrix satifies Jj = I, which gives eΣj = Σj. Thus following (Zwicker et al., 2002; 2001)
we have
bIi(r) =
M
X
j=1
ρj(2π)
3
2
eΣj

1
2 Z
1
(2π)
3
2
eΣj

1
2
exp

−1
2 (x −epj)⊤eΣ−1
i
(x −pj)

dz
(A.9)
=
M
X
j=1
ρj(2π)
3
2 |Σi|
1
2
Z
1
(2π)
3
2 |Σi|
1
2
exp

−1
2 (x −pi)⊤Σ−1
i
(x −pi)

dz
(A.10)
=
M
X
j=1
ρj
1
2π
bΣj
1/2 exp

−1
2 (bx −bpj)⊤bΣ−1
j (bx −bpj)

.
(A.11)
A.3
Description of Datasets
The characteristics of datasets used in this paper are summarized as follows:
• EMPIAR-10005 (TRPV1): A well-characterized protein in the vertebrate TRP family, and is
frequently used to investigate fundamental TRP channel functions and structures.
• EMPIAR-10028 (Pf 80S ribosome): Contains a complex composition of the antibiotic and Pf
ribosome and is commonly used for evaluation of the reconstruction resolution.
• EMPIAR-10049 (Synaptic RAG1–RAG2 complex): Exhibits substantial compositional and
conformational heterogeneity and is therefore challenging for high-resolution reconstruction.
• EMPIAR-10076 (L17-depleted 50S ribosomal intermediates): Comprises complex intermediate
15

<!-- page 16 -->
GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction
assembly states that can cause reconstruction quality degradation.
16
