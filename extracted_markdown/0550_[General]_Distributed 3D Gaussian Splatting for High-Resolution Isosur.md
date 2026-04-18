<!-- page 1 -->
Distributed 3D Gaussian Splatting for
High-Resolution Isosurface Visualization
1st Mengjiao Han
Argonne National Laboratory
Lemont, IL, USA
hanm@anl.gov
2nd Andres Sewell
Utah State University
Logan, UT, USA
a02024444@usu.edu
3rd Joseph Insley
Argonne National Laboratory
Lemont, IL, USA
insley@anl.gov
4th Janet Knowles
Argonne National Laboratory
Lemont, IL, USA
jknowles@anl.gov
5th Victor A. Mateevitsi
Argonne National Laboratory
University of Illinois Chicago
Lemont, IL, USA
vmateevitsi@anl.gov
6th Michael E. Papka
Argonne National Laboratory
University of Illinois Chicago
Lemont, IL, USA
papka@anl.gov
7th Steve Petruzza
Utah State University
Logan, UT, USA
steve.petruzza@usu.edu
8th Silvio Rizzi
Argonne National Laboratory
Lemont, IL, USA
srizzi@anl.gov
Abstract—3D
Gaussian
Splatting
(3D-GS)
has
recently
emerged as a powerful technique for real-time, photorealistic ren-
dering by optimizing anisotropic Gaussian primitives from view-
dependent images. While 3D-GS has been extended to scientific
visualization, prior work remains limited to single-GPU settings,
restricting scalability for large datasets on high-performance
computing (HPC) systems. We present a distributed 3D-GS
pipeline tailored for HPC. Our approach partitions data across
nodes, trains Gaussian splats in parallel using multi-nodes and
multi-GPUs, and merges splats for global rendering. To eliminate
artifacts, we add ghost cells at partition boundaries and apply
background masks to remove irrelevant pixels. Benchmarks on
the Richtmyer–Meshkov datasets (about 106.7M Gaussians) show
up to 3X speedup across 8 nodes on Polaris while preserving
image quality. These results demonstrate that distributed 3D-GS
enables scalable visualization of large-scale scientific data and
provide a foundation for future in situ applications.
Index Terms—3D Gaussian Splatting, Distributed 3D-GS, Sci-
entific Data Visualization
I. INTRODUCTION
3D Gaussian Splatting (3D-GS) [1] is a recent technique
that enables photorealistic, real-time rendering of complex 3D
scenes by optimizing anisotropic Gaussian primitives from
view-dependent images. Unlike neural implicit methods such
as NeRF, 3D-GS avoids the need for a neural network forward
pass at inference, making it significantly faster while maintain-
ing high visual fidelity. Recent studies have applied 3D-GS
to scientific visualization [2]–[5], but pipelines remain single-
GPU, limiting scalability for large datasets. These datasets of-
ten exceed the memory of one GPU and are distributed across
compute nodes on HPC, making centralized training impracti-
cal. We address this gap with a distributed 3D-GS pipeline
that supports multi-node, multi-GPU execution. Each node
processes its local data subset, trains splats independently,
and results are merged for global rendering. Additionally, we
incorporate ghost cells and background masking to address
rendering artifacts such as gaps and white streaks. The source
code can be found at https://github.com/MengjiaoH/Grendel-
GS-SciVIS.
Our contributions are as follows.
• Distributed 3D-GS for HPC: A multi-node, multi-GPU
pipeline for large-scale scientific visualization, designed
for datasets partitioned across HPC nodes.
• Scalability Benchmarks: Performance evaluation across
multiple datasets, image resolutions, and node counts.
• Foundation for In Situ Visualization: A first step toward
scalable in situ workflows using 3D-GS for scientific data
visualization.
II. METHOD
Our distributed workflow (Figure 1) proceeds as follows:
• Isosurface Extraction: Use ParaView1 to extract isosur-
face point clouds from volume datasets as initial Gaussian
primitives.
• Camera Setup: Generate a structured orbital set of
synthetic camera views. All nodes use identical settings
for training consistency.
• Data Partitioning: The dataset is divided into n parti-
tions, one for each compute node, add ghost cells to avoid
rendering gaps at partition boundaries.
• Image Rendering and Masking: On each node, render
images and background masks for its own data partition.
These masks prevent irrelevant splats and white streaks.
• Parallel Training: Each node trains a Gaussian splat-
ting model independently using multi-GPU training ap-
proach [6].
• Global
Reconstruction: Splats from all nodes are
merged for final rendering.
This design removes the need to centralize large datasets,
making distributed training tractable on HPC systems.
III. RESULTS
We evaluated on three datasets: Kingsnake (110.3 MB,
about 4M points)2, Rayleigh–Taylor instability (491 MB,
1https://www.paraview.org
2https://www.digimorph.org/index.phtml
arXiv:2509.12138v1  [cs.DC]  15 Sep 2025

<!-- page 2 -->
Fig. 1: Workflow of our distributed 3D-GS pipeline for large-
scale isosurface visualization on HPC systems.
about 18.2M points) [7] and Richtmyer-Meshkov instability
(5.3 GB, about 106.7M points) [8]. Training was performed
on Polaris at Argonne3, with 4 NVIDIA A100 GPUs per node
and 448 training images per dataset.
A. Rendering Improvements
Without our adjustments, merging splats across nodes in-
troduced gaps and white streak artifacts (Figure 4b). Adding
ghost cells ensured smooth partition boundaries, while back-
ground masks removed unnecessary background splats, yield-
ing artifact-free results (Figure 2c).
(a) Ground Truth
(b) W/O GC and
Masks
(c) Our Method W/
GC and Masks
Fig. 2: Visualization comparison using the Kingsnake scan
dataset. (a) Ground truth, (b) rendering without ghost cells
(GC) or background masks, and (c) our method with GC and
background masks.
B. Scaling Efficiency
1) Single Node Scaling:
On smaller datasets such as
Kingsnake and Rayleigh–Taylor, multi-GPUs reduced training
time significantly (Table I). For example, at 20482 resolution,
Kingsnake achieved a 5.6X speedup with 4 GPUs vs. 1,
confirming effective intra-node scaling. Since one A100 GPU
can handle only 11.2M Gaussians [6], multi-GPU training was
essential for larger datasets like Rayleigh–Taylor (Table I).
Visualizations confirm high reconstruction quality (Figure 3),
with detailed metrics in Tables II and III.
3https://www.alcf.anl.gov/polaris
TABLE I: Training time (minutes) for Kingsnake and
Rayleigh–Taylor at different resolutions and GPU counts. ‘X’
marks runs that exceeded a single A100 GPU’s memory.
Kingsnake (∼4M)
Rayleigh–Taylor (∼18M)
#GPUs
1024
2048
1024
2048
1
18.60
48.00
X
X
2
10.48
15.46
21.88
50.10
4
5.97
8.50
12.20
16.84
TABLE II: PSNR (↑), SSIM (↑), and LPIPS (↓) for the
Kingsnake dataset across different image resolutions and GPU
counts.
512
1024
2048
#GPUs
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
1
25.52
0.95
0.056
26.90
0.96
0.056
25.12
0.93
0.089
2
25.87
0.96
0.046
28.63
0.97
0.035
29.33
0.97
0.030
4
25.87
0.96
0.046
25.03
0.93
0.067
29.32
0.97
0.030
(a) Ground Truth
(b) Distributed 3D-GS
Fig. 3: Visualization on the Rayleigh–Taylor dataset: (a)
ground truth and (b) reconstruction with our distributed 3D-
GS method, achieving high fidelity (PSNR = 36.37, SSIM =
0.9905, LPIPS = 0.011).
2) Multi-Node Scaling: To evaluate scalability, we bench-
marked our distributed 3D-GS pipeline on the Rayleigh–Taylor
and Richtmyer–Meshkov datasets using multi-node runs with
4 GPUs per node. For Rayleigh–Taylor, training scaled effi-
ciently with a 1.4× speedup from 2 to 4 nodes (Table IV),
while maintaining high quality (Table V). However, because
the dataset is not large enough, scaling from 4 to 8 nodes
provided only limited gains in training time. The larger Richt-
myer–Meshkov dataset required ≥4 nodes due to memory
limits, but scaling to 8 nodes achieved a 3.1X speedup at 20482
(Table IV), with stable image quality (Table VI) and visual-
izations nearly identical to ground truth (Figure 4). Overall,
these results confirm that distributed 3D-GS scales effectively
across multiple nodes, providing significant speedups while
preserving visualization fidelity for both moderate and large-
TABLE III: PSNR (↑), SSIM (↑), and LPIPS (↓) for the
Rayleigh–Taylor dataset across different image resolutions and
GPU counts.
512
1024
2048
#GPUs
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
2
31.62
0.99
0.014
34.21
0.99
0.010
36.30
0.99
0.011
4
31.63
0.99
0.014
34.22
0.99
0.010
36.37
0.99
0.011

<!-- page 3 -->
TABLE IV: Training time (minutes) for Rayleigh–Taylor and
Richtmyer–Meshkov at different resolutions and node counts.
‘X’ = run failed due to memory limits.
Rayleigh–Taylor (∼18M)
Richtmyer-Meshkov (∼106.7M)
#Nodes
1024
2048
1024
2048
2
7.22
11.97
X
X
4
5.08
8.33
10.07
32.03
8
5.30
7.45
7.87
10.18
TABLE V: PSNR (↑), SSIM (↑), and LPIPS (↓) for the
Rayleigh–Taylor dataset across different image resolutions and
compute node counts.
1024x1024
2048x2048
#Nodes
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
2
33.64
0.99
0.012
37.22
0.99
0.008
4
33.61
0.99
0.012
37.15
0.99
0.008
8
33.64
0.99
0.012
35.42
0.99
0.012
TABLE VI: PSNR (↑), SSIM (↑), and LPIPS (↓) for the
Richtmyer-Meshkov dataset across different image resolutions
and compute node counts.
1024x1024
2048x2048
#Nodes
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
4
28.46
0.97
0.018
30.04
0.97
0.019
8
28.20
0.96
0.019
30.04
0.97
0.019
scale datasets.
IV. CONCLUSION AND FUTURE WORK
We introduced a distributed 3D-GS pipeline for large-
scale scientific visualization on HPC. By partitioning datasets
across nodes and incorporating ghost cells and background
masks, we eliminate rendering artifacts while enabling multi-
node, multi-GPU scaling. Our results show up to 3× faster
(a) Ground Truth
(b) Distributed 3D-GS Result
Fig. 4: Visualization of the Richtmyer–Meshkov instability
dataset [8] containing 106.7M Gaussians. (a) Ground truth im-
age rendered directly from the point cloud. (b) Reconstruction
from our distributed 3D-GS method at 2048×2048 resolution.
Training was performed on 8 Polaris nodes at Argonne, each
with 4 NVIDIA A100 GPUs, completing in 8 minutes. The
reconstruction achieves high quality: PSNR=30, SSIM=0.97,
LPIPS=0.019.
training with consistently high quality. Future work will ex-
tend this framework to real large-scale datasets, explore load
balancing strategies, integrate with simulation pipelines for
in situ rendering, and develop uncertainty quantification to
provide scientists with confidence measures in reconstructed
visualizations.
ACKNOWLEDGMENT
This research used resources of the Argonne Leadership
Computing Facility, a U.S. Department of Energy (DOE)
Office of Science user facility at Argonne National Laboratory
and is based on research supported by the U.S. DOE Office
of Science-Advanced Scientific Computing Research Program,
under Contract No. DE-AC02-06CH11357.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-
sam.inria.fr/fungraph/3d-gaussian-splatting/
[2] K. Ai, K. Tang, and C. Wang, “Nli4volvis: Natural language interaction
for volume visualization via llm multi-agents and editable 3d gaussian
splatting,” arXiv preprint arXiv:2507.12621, 2025.
[3] K. Tang, S. Yao, and C. Wang, “ivr-gs: Inverse volume rendering
for explorable visualization via editable 3d gaussian splatting,” IEEE
Transactions on Visualization and Computer Graphics, 2025.
[4] A. Sewell, L. Dyken, V. A. Mateevitsi, W. Usher, J. Amstutz, T. Marrinan,
K. Reda, S. Rizzi, M. E. Papka, S. Kumar et al., “High-quality approxi-
mation of scientific data using 3d gaussian splatting,” in 2024 IEEE 14th
Symposium on Large Data Analysis and Visualization (LDAV).
IEEE,
2024, pp. 73–74.
[5] S. Yao and C. Wang, “Volseggs: Segmentation and tracking in dy-
namic volumetric scenes via deformable 3d gaussians,” arXiv preprint
arXiv:2507.12667, 2025.
[6] H. Zhao, H. Weng, D. Lu, A. Li, J. Li, A. Panda, and S. Xie, “On scaling
up 3d gaussian splatting training,” in European Conference on Computer
Vision.
Springer, 2024, pp. 14–36.
[7] A. W. Cook, W. Cabot, and P. L. Miller, “The mixing transition in
Rayleigh-Taylor instability,” Journal of Fluid Mechanics, vol. 511, pp.
333–362, 2004.
[8] R. H. Cohen, W. P. Dannevik, A. M. Dimits, D. E. Eliason, A. A.
Mirin, Y. Zhou, D. H. Porter, and P. R. Woodward, “Three-dimensional
simulation of a richtmyer–meshkov instability with a two-scale initial
perturbation,” Physics of Fluids, vol. 14, no. 10, pp. 3692–3709, oct 2002.
