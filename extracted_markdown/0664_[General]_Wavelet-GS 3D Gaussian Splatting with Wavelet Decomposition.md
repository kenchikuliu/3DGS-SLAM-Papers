<!-- page 1 -->
Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition
Beizhen Zhao
bzhao610@connect.hkust-gz.edu.cn
AI Thrust, HKUST(GZ)
Yifan Zhou
AI Thrust, HKUST(GZ)
Sicheng Yu
AI Thrust, HKUST(GZ)
Zijian Wang
AI Thrust, HKUST(GZ)
Hao Wang*
haowang@hkust-gz.edu.cn
AI Thrust, HKUST(GZ)
Figure 1: Overview of our model. Through 3D wavelet decomposition, the 3D point cloud is divided into low frequency and high
frequency components. These two parts are trained through their own customized strategies and finally integrated through
rendering to obtain the final novel views.
Abstract
3D Gaussian Splatting (3DGS) has revolutionized 3D scene recon-
struction, which effectively balances rendering quality, efficiency,
and speed. However, existing 3DGS approaches usually generate
plausible outputs and face significant challenges in complex scene
reconstruction, manifesting as incomplete holistic structural out-
lines and unclear local lighting effects. To address these issues
simultaneously, we propose a novel decoupled optimization frame-
work, which integrates wavelet decomposition into 3D Gaussian
Splatting and 2D sampling. Technically, through 3D wavelet decom-
position, our approach divides point clouds into high-frequency
and low-frequency components, enabling targeted optimization
for each. The low-frequency component captures global structural
outlines and manages the distribution of Gaussians through vox-
elization. In contrast, the high-frequency component restores in-
tricate geometric and textural details while incorporating a relight
module to mitigate lighting artifacts and enhance photorealistic
rendering. Additionally, a 2D wavelet decomposition is applied to
the training images, simulating radiance variations. This provides
critical guidance for high-frequency detail reconstruction, ensuring
seamless integration of details with the global structure. Exten-
sive experiments on challenging datasets demonstrate our method
* Corresponding author.
achieves state-of-the-art performance across various metrics, sur-
passing existing approaches and advancing the field of 3D scene
reconstruction.
CCS Concepts
• Computing methodologies →Computer vision; Rendering;
Shape modeling.
Keywords
3D Reconstruction, Gaussian Splatting, Wavelet Transformation,
Point based rendering
1
Introduction
Reconstructing high-fidelity 3D scenes remains a fundamental chal-
lenge in computer vision and graphics, driven by its importance in
applications such as virtual reality, autonomous driving, and cul-
tural heritage digitization [43, 47, 48]. While traditional multi-view
stereo methods [9, 12, 34] and modern neural implicit represen-
tations [5, 6, 28] have made significant progress, the complexity
of outdoor environments - characterized by intricate geometries,
detailed textures, and dynamic lighting - continues to challenge the
limits of current methods [2, 16, 33].
Among recent advancements, 3D Gaussian Splatting (3DGS)
[4, 14, 40, 44] has garnered significant attention for its ability to
arXiv:2507.12498v2  [cs.GR]  21 Jul 2025

<!-- page 2 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, and Hao Wang*
balance rendering quality, training speed, and real-time perfor-
mance [29, 32]. However, existing 3DGS-based approaches [3, 10,
15, 20, 24, 45, 49] primarily produce plausible results, falling short
of achieving high-fidelity reconstructions. These limitations are
particularly pronounced in holistic structural alignment and the
accurate modeling of local lighting effects [35]. The reliance on
fitting primitives to 2D image distributions often results in a fail-
ure to align with real-world scene structures, reducing robustness
in novel view synthesis, particularly under sparse training views.
Additionally, the absence of explicit mechanisms for modeling high-
frequency details and handling complex lighting further constrains
their effectiveness in generating photorealistic reconstructions.
To address these challenges, we propose a novel framework that
combines wavelet decomposition with 3DGS, marking the first
attempt to incorporate wavelet transforms into 3DGS. By decom-
posing 3D point clouds into low-frequency and high-frequency
components, our method decouples the optimization process for
tailored reconstruction strategies. The low-frequency component
captures global structural outlines, while the high-frequency com-
ponent restores intricate details, ensuring improved fidelity.
Specifically, the low-frequency component primarily concen-
trates on the outline of the global scene structure, ensuring the
overall coherence of the scene by effectively managing the distri-
bution of Gaussians through voxelization. We further propose a
gradient and opacity based training strategy to enhance the struc-
tural representation of the low-frequency component. To address
lighting artifacts, we introduce a relight module within the high-
frequency component to explicitly model lighting variations, en-
abling realistic color rendering and improving robustness under
varying illumination conditions.
Furthermore, we extend our framework with a 2D wavelet de-
composition to model the structural features and contrast between
light and dark, which serves as a foundation for subsequent pho-
torealistic color recovery. By combining the 3D and 2D wavelet
decomposition techniques, we achieve a comprehensive framework
that balances global coherence and fine detail restoration while
maintaining photorealistic visual quality.
Extensive experiments on challenging datasets, including Waymo
[37], MipNeRF360 [1], Tanks&Temples [17] and JHU-Drone [19],
demonstrate that our method achieves significant improvements
over state-of-the-art approaches. By integrating wavelet decompo-
sition into the 3DGS pipeline, we achieve a scalable and robust so-
lution for various 3D scenes, surpassing state-of-the-art techniques
in structural accuracy, detail preservation, and visual realism.
In summary, our contributions are as follows:
• We propose the first framework that integrates wavelet de-
composition with 3DGS, which consists of 3D decomposition
on point clouds and 2D sampling for structural features sim-
ulation and relight.
• We propose distinct customized optimization strategies for
high- and low-frequency components, enabling robust and
scalable scene modeling.
• Experiment results demonstrate that our framework sur-
passes existing state-of-the-art results on all four 3D datasets.
2
Related Work
2.1
Gaussian Splatting Based Variants
3DGS [14] has emerged as a state-of-the-art method, offering high
visual fidelity, efficient training, and real-time rendering through a
primitive-based representation. Unlike implicit field-based methods,
3DGS allows for flexible camera paths and dynamic allocation of
representational capacity [11, 41, 46].
Variants of 3DGS, such as 2DGS [10], GOF [45], and PGSR [3],
have extended its application scope while addressing specific chal-
lenges. GOF [45] integrates geometric and optical flow constraints,
improving temporal consistency in dynamic scenes, but its effective-
ness diminishes in sparse viewpoints situations. PGSR [3] employs a
progressive refinement process to achieve high-quality detail recov-
ery, yet is limited in its ability to handle dynamic high-complexity
outdoor environments. 2DGS [10] focuses on optimizing splats in
the image plane, providing a simpler framework, though it sacrifices
flexibility in aligning to complex 3D geometries.
A shared limitation among these methods is the misalignment
between Gaussian distributions and the underlying 3D scene struc-
ture. Instead of directly modeling scene geometry, primitives are
often trained to fit 2D image distributions, reducing robustness in
novel view synthesis, particularly in cases with sparse or incom-
plete training data.
2.2
Voxel-Based Representations
Voxelization based approaches have been developed to improve
scalability and structural coherence [18, 23, 36]. Scaffold-GS [22]
introduces a voxelization Gaussian framework to provide a robust
initialization of the 3D points. This strategy enhances the overall
alignment of geometric features but lacks explicit mechanisms for
recovering high-frequency details. Building on this, Octree-GS [31]
incorporates an octree structure to optimize memory usage and im-
prove efficiency. However, both methods struggle with fine-grained
detail preservation and often produce oversmoothed reconstruc-
tions in regions with complex textures or intricate geometries.
While Gaussian-based and hierarchical voxel-based methods
provide valuable contributions to 3D reconstruction, their limita-
tions in scalability and detail preservation still remain challenges
to solve. Our work addresses these challenges by integrating a
wavelet-based decomposition with hierarchical Gaussian represen-
tations, offering a scalable and high-fidelity solution for unbounded
scene reconstruction.
3
Methodology
3.1
Overview
Our proposed methodology is structured into three key compo-
nents:
(1) 3D Wavelet Decomposition: We decompose the input point
cloud into low-frequency and high-frequency components. The low-
frequency component captures the scene’s structural framework,
while the high-frequency component preserves fine details. This
decomposition allows targeted optimization for different aspects of
reconstruction.
(2) 2D Wavelet Decomposition: By applying 2D wavelet decom-
position to structural features, we capture radiance variations across

<!-- page 3 -->
Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 2: Framework of Wavelet-GS. We begin by preprocessing the point cloud by voxelization and 3D wavelet decomposition.
Then we train the low freq component and the high freq component in the same time, which utilize individual optimization
strategy according to the value of training gradint and deviation of high freq component to update points. A relight module
to compensate the color changes in high freq compoment override the color of final gaussians and achieve photorealistic
reconstructions.
scales for subsequent modeling of lighting effects, addressing arti-
facts and enhancing photorealistic rendering.
(3) Individual Training Strategies: Tailored optimization strate-
gies are applied to each frequency component. The low-frequency
component focuses on global coherence using a dynamic Gaussian
growth method, while the high-frequency component emphasizes
detail restoration and radiance modeling, ensuring efficiency and
scalability.
By balancing global coherence and fine-detail accuracy, our
methodology offers a robust and efficient solution for complex
3D scene reconstruction. The overall pipeline is shown in Fig. 2.
3.2
Preliminaries
3.2.1
Wavelet decomposition. Wavelet decomposition is a versatile
mathematical technique for analyzing and representing signals
or functions across different levels of detail [26, 38, 42]. Unlike
the Fourier transform, which focuses solely on global frequency
characteristics, wavelet decomposition provides a combined view
of spatial and frequency information [25, 27].
In general, a function 𝑓(𝑥) can be expressed in terms of wavelet
basis functions 𝜓𝑗,𝑘(𝑥) as:
𝑓(𝑥) =
∑︁
𝑗∈Z
∑︁
𝑘∈Z
𝑐𝑗,𝑘𝜓𝑗,𝑘(𝑥),
(1)
where𝜓𝑗,𝑘(𝑥) is derived by scaling and translating a mother wavelet
𝜓(𝑥), defined as:
𝜓𝑗,𝑘(𝑥) = 2𝑗/2𝜓(2𝑗𝑥−𝑘),
(2)
where 𝑗controls the scale, 𝑘specifies the translation, and 𝑐𝑗,𝑘are
the corresponding wavelet coefficients.
One of the most important mathematical properties of wavelet
decomposition is its linearity, which allows the superposition of
wavelet coefficients [7]. Given two signals 𝑓1(𝑥) and 𝑓2(𝑥), their
wavelet transforms satisfy:
𝑇𝜓{𝑓1(𝑥) + 𝑓2(𝑥)} = 𝑇𝜓{𝑓1(𝑥)} +𝑇𝜓{𝑓2(𝑥)},
(3)
where 𝑇𝜓denotes the wavelet transform operator. This additivity
property is critical for applications in computer graphics, such as
combining multiple levels of detail in texture synthesis or blending
multiple lighting effects in rendering.
3.2.2
Radiance Transfer. Radiance transfer plays a critical role in
rendering algorithms by characterizing how light interacts with
surfaces in a scene [8]. At its core lies the rendering equation:
𝐿(𝑥,𝜔𝑜) =
∫
Ω
𝑓𝑟(𝑥,𝜔𝑜,𝜔𝑖)𝐿𝑖(𝑥,𝜔𝑖)𝐷(𝜔𝑖· 𝑛) 𝑑𝜔𝑖,
(4)
where 𝐿(𝑥,𝜔𝑜) represents the outgoing radiance at point 𝑥in di-
rection 𝜔𝑜, 𝑓𝑟is the bidirectional reflectance distribution function
(BRDF), 𝐿𝑖(𝑥,𝜔𝑖) denotes the incident radiance from direction 𝜔𝑖,
and 𝐷(𝜔𝑖· 𝑛) accounts for the geometric attenuation due to the
surface normal 𝑛. Here, Ω denotes the upper hemisphere centered
around the surface normal 𝑛.
To simplify computations, the diffuse BRDF model assumes
isotropic reflection, making lighting view independent [30]. This
reduces the rendering equation to:
𝐿𝐷(𝑥) = 𝜌(𝑥)
𝜋
∫
Ω
𝐿𝑖(𝑥,𝜔𝑖) max(𝜔𝑖· 𝑛, 0) 𝑑𝜔𝑖,
(5)

<!-- page 4 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, and Hao Wang*
where 𝜌(𝑥) is the surface albedo, and 𝐿𝐷(𝑥) is the outgoing diffuse
radiance.
In practical scenarios involving omnidirectional illumination,
the incident radiance 𝐿𝑖and the cosine-weighted transfer term
max(𝜔𝑖·𝑛, 0) can be approximated using Spherical Harmonics (SH)
expansions. This transforms the integral formulation of diffuse
radiance into a simple inner product:
𝐿𝑆(𝑥) = 𝜌(𝑥)
𝜋
l · d,
(6)
where l and d are the SH coefficient vectors of the incident radiance
and transfer term, respectively. This representation leverages the
orthogonality of SH bases, allowing for an efficient and compact
computation of the radiance transfer.
This formulation serves as the theoretical foundation for the
lighting modeling techniques used later in our framework, par-
ticularly in enhancing the realism of high-frequency component
rendering.
3.2.3
3D Gaussian Splatting. 3D Gaussian Splatting (3DGS) [14] is
a recent real-time rendering technique that represents a 3D scene
as a collection of anisotropic Gaussian primitives. Each Gaussian is
parameterized by a center position 𝜇, orientation 𝑟, scale 𝑠, color 𝑐,
and opacity 𝛼. During rendering, Gaussians are projected onto the
image plane and contributes to the pixel color via alpha blending.
The color at a pixel 𝑥′ is computed as:
𝐶(𝑥′) =
∑︁
𝑖∈𝑁
𝑐𝑖𝛼′
𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼′
𝑗)
(7)
where 𝑁is the set of Gaussians influencing 𝑥′, and 𝛼′
𝑖denotes
the opacity after weighting by a 2D projected Gaussian density
function.
In the original 3DGS pipeline, the Gaussians are initialized from
a sparse point cloud reconstructed by COLMAP, and their parame-
ters are optimized by minimizing the photometric reconstruction
loss between rendered and ground-truth images. In this work, we
propose a novel training framework for 3DGS by integrating multi-
scale frequency-aware optimization into the pipeline.
3.3
3D Wavelet Decomposition
High-fidelity 3D scene reconstruction often requires balancing
global structural accuracy and detailed spatial fidelity. To achieve
this, we propose a 3D wavelet-based point cloud decomposition
framework that explicitly decomposes the 3D point cloud P into
low-frequency and high-frequency components. This decomposi-
tion allows targeted optimization for different aspects of the recon-
struction, addressing the limitations of existing methods that lack
explicit mechanisms for handling complex geometries and lighting.
The discrete wavelet transform (DWT) is performed along the XYZ
spatial dimensions:
P
DWT
−−−−→{Plow, Phigh},
(8)
where Plow captures global structural information and Phigh en-
codes fine-grained details. By isolating these components, we miti-
gate issues such as over-smoothing in low-frequency representa-
tions and instability in high-frequency reconstructions.
Figure 3: 3D wavelet decomposition. The input data for the
initialization of the 3D gaussians would be a point cloud,
which provides xyz information in the real world. The orig-
inal 3D points are divided into two parts which represents
low frequency component and high frequency component.
For a single-level discrete wavelet transform (DWT), the decom-
position is expressed as:
Plow(𝑖, 𝑗,𝑘) =
∑︁
𝑚,𝑛,𝑝
P(𝑚,𝑛, 𝑝)𝜙𝑖,𝑗,𝑘(𝑚,𝑛, 𝑝),
(9)
Phigh(𝑖, 𝑗,𝑘) =
∑︁
𝑚,𝑛,𝑝
P(𝑚,𝑛, 𝑝)𝜓𝑖,𝑗,𝑘(𝑚,𝑛, 𝑝),
(10)
where 𝜙𝑖,𝑗,𝑘(𝑚,𝑛, 𝑝) is the scaling function and𝜓𝑖,𝑗,𝑘(𝑚,𝑛, 𝑝) is the
wavelet function.
Because of the linear additivity of wavelet transform, the inverse
wavelet transform allows reconstruction of the original point cloud
by combining Plow and Phigh:
P(𝑖, 𝑗,𝑘) = Plow(𝑖, 𝑗,𝑘) + Phigh(𝑖, 𝑗,𝑘).
(11)
The resulting decomposed point cloud branches Plow and Phigh
are voxelized into Vlow and Vhigh respectively, which are used to
generate gaussians through optimization stategy and light network.
3.4
2D Wavelet Decomposition
To simulate complex interactions between geometry and environ-
mental lighting, we apply 2D DWT, which decomposes images into
hierarchical frequency components. This rich radiance information
is used to guide high-frequency detail reconstruction, effectively
addressing lighting artifacts and ensuring seamless integration with
the global scene structure:
𝐼gray(𝑥,𝑦)
DWT
−−−−→{𝐴, 𝐻𝐿,𝑉𝐿, 𝐷𝐿},
(12)
where 𝐴is the approximation map, and 𝐻𝐿, 𝑉𝐿, and 𝐷𝐿are the
horizontal, vertical, and diagonal detail maps at level 𝐿. Then we
simulate the structural features 𝑀through a MLP:
𝑀= MLP(𝐴),
(13)
The resulting structural feature 𝑀is passed into the radiance
transfer modeling stage to approximate the global light intensity
effects. The structural feature is represented using spherical har-
monics (SH) up to the second degree (𝑛= 2):
𝐿𝑖(x, 𝝎𝑖) =
𝑛
∑︁
𝑙=0
𝑙∑︁
𝑚=−𝑙
𝑐𝑙𝑚𝑌𝑙𝑚(𝝎𝑖),
(14)
where 𝑐𝑙𝑚are the SH coefficients, and 𝑌𝑙𝑚are the SH basis func-
tions. This compact representation allows efficient computation of
radiance transfer. we follow [13] and use LSH-env:

<!-- page 5 -->
Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 4: Simulation for the structural feature We use 2D
wavelet transform to simulate the structural features by de-
composing the images used for training. Through this func-
tion, we focus on the tensity of light and shadow to achieve
more realistic render effect.
LSH = E

min(0, 𝐿𝑖(𝑀)2
,
(15)
3.5
Individual Optimization Strategy
This section details the Individual Optimization Strategy employed
in our work. We design an individual optimization strategy, allow-
ing for tailored optimization of different frequency components.
We manage Gaussians based on voxel x, and fuse high-frequency
and low-frequency Gaussians by leveraging the linearity of wavelet
transform:
G(x) = G𝑙(x) + Gℎ(x).
(16)
3.5.1
Low-Frequency Component. The low-frequency component
focuses on capturing the global structural framework of the scene.
After voxelizing Plow into a grid Vlow, each voxel is parameterized
by Gaussian functions predicted through a multi-layer perceptron
(MLP):
G𝑙(x) =
n
𝐹𝜎
𝑙(x𝑙), 𝐹𝜇
𝑙(x𝑙), 𝐹𝛼
𝑙(x𝑙), 𝐹𝑐
𝑙(x𝑙)
o
,
(17)
where G𝑙denotes the Gaussian representation of the low-frequency
component. 𝐹𝜎
𝑙, 𝐹𝜇
𝑙, 𝐹𝛼
𝑙and 𝐹𝑐
𝑙represent the MLP network to gen-
erate variance 𝜎𝑙, mean 𝜇𝑙, opacity 𝛼and color 𝑐of the gaussians.
Inspired by [22], we design a grow-and-prune strategy for voxel
centers to provide robust global structure guidance. For each voxel,
the average gradient of the included neural gaussians is computed.
To further regulate the addition of new voxels, we apply a random
picking strategy, effectively controlling the expansion rate of voxels.
For pruning, the opacity values of neural gaussians associated with
each voxel are accumulated. voxels that fail to achieve sufficient
opacity are removed from the scene to eliminate trivial points:
x𝑙←x𝑙+ 𝜂𝑙
𝜕L
𝜕x𝑙
,
(18)
where 𝜂𝑙is the learning rate. This strategy dynamically adjusts
voxel positions, improving alignment with the global geometry.
For pixel-level rendering results, we employ L1 and Structural
Similarity Index Measure (SSIM) loss functions [39] LSSIM and
incorporate a volume regularization term [21] Lvol to supervise
the accuracy of rendered outputs.
Lpixel = 𝜆1L1 + 𝜆SSIMLSSIM + 𝜆volLvol,
(19)
where the volume regularization Lvol is:
Lvol =
𝑁ng
∑︁
𝑖=1
Prod(𝑠𝑖).
(20)
Here, 𝑁ng denotes the number of neural gaussians in the scene
and Prod(·) is the product of the scale values 𝑠𝑖of each neural
Gaussian. The volume regularization term encourages the neural
gaussians to be small with minimal overlapping.
3.5.2
High-Frequency Component. The high-frequency component
restores intricate geometric and textural details such as sharp edges
and texture variations. Similarly, the high-frequency component
Phigh is voxelized into Vhigh, and its corresponding Gaussian pa-
rameters are predicted using another set of MLP:
Gℎ(x) =
n
𝐹𝜎
ℎ(xℎ), 𝐹𝜇
ℎ(xℎ), 𝐹𝛼
ℎ(xℎ), 𝜎(𝐹𝑐
ℎ(xℎ) + ˜𝑐𝑘)
o
,
(21)
where Gℎdenotes the Gaussian representation of the high-frequency
component. 𝐹𝜎
ℎ, 𝐹𝜇
ℎ, 𝐹𝛼
ℎand 𝐹𝑐
ℎrepresent the MLP network to gen-
erate variance 𝜎ℎ, mean 𝜇ℎ, opacity 𝛼and color 𝑐of the gaussians.
𝜎is sigmoid function and ˜𝑐𝑘is the relight color from the relight
module.
To address the challenges posed by high-frequency components
in our data, we designed a strategy that focuses on mitigating the
growth of voxels associated with extreme high-frequency devia-
tions. Specifically, we apply a thresholding technique to segment
the high-frequency deviation values, identifying those that are ei-
ther excessively large or small. By generating a mask based on these
thresholds, we effectively reduce the growth of voxels in regions
where the high-frequency deviations fall outside the desired range.
To reconstruct the structural details preserved across multiple
scales and improve the visual quality of the aligned images, we
design a Laplacian-Wavelet loss which measures the structural sim-
ilarity between two images across multiple scales by constructing
their respective Laplacian pyramids and 2D wavelet transformation
results.
LL-W =
𝐿
∑︁
𝑙=1
∥L(𝑙)
1
−L(𝑙)
2 ∥1 +
𝑊
∑︁
𝑤=1
∥W (𝑤)
1
−W (𝑤)
2
∥1
(22)
where 𝐿denotes the number of levels in the Laplacian pyramid.
L(𝑙)
1
and L(𝑙)
2
are the Laplacian pyramid representations of images
𝐼1 and 𝐼2 at level 𝑙respectively, and W (𝑤)
1
and W (𝑤)
2
are the 2D
DWT results of images 𝐼1 and 𝐼2 at level 𝑤.
To achieve more realistic color and photorealistic rendering, we
address the challenges of dynamic lighting and shadowing through
a relight module. To stabilize training, we employ a two-stage
strategy inspired by [13].
In the warm-up stage, we focus on unshadowed radiance transfer
to learn basic albedo and illumination, where the visibility function
𝐷(x, 𝝎𝑖) = 1. Here, x denotes the surface point, 𝝎𝑖is the incident
light direction, and 𝐷(x, 𝝎𝑖) represents the visibility function that

<!-- page 6 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, and Hao Wang*
accounts for shadowing effects. Then we use the structural features
from the 2D wavelet transform to calculate the color ˜𝑐𝑘:
˜𝑐𝑘= 𝝆𝑘⊙n𝑡
𝑘𝑀envn𝑘,
(23)
where 𝝆𝑘∈R3 represents the diffuse albedo of the𝑘-th Gaussian,
n𝑘is the normal vector of the 𝑘-th Gaussian, 𝑀env is the environ-
ment map, and ⊙denotes element-wise multiplication.
In the relight stage, we incorporate shadowing effects by learning
the visibility function 𝐷(x, 𝝎𝑖) through an additional MLP. The
visibility function 𝐷(x, 𝝎𝑖) is parameterized by spherical harmonics
coefficients d𝑘, where 𝑘indexes the Gaussian components. Finally,
the relight color ˜𝑐𝑘for each Gaussian is computed as:
˜𝑐𝑘= 𝝆𝑘⊙
(𝑛+1)2
∑︁
𝑖=1
𝑀𝑖
env · d𝑖
𝑘,
(24)
where 𝑀𝑖env represents the 𝑖-th component of the environment
map, d𝑖
𝑘is the 𝑖-th spherical harmonics coefficient for the 𝑘-th
Gaussian, and 𝑛is the order of the spherical harmonics.
In summary, our final training loss L consists of the pixel-based
loss Lpixel, the SH environment loss LSH-env and the Laplacian-
Wavelet loss LL-W:
L = Lpixel + 𝜆SHLSH + 𝜆L-WLL-W,
(25)
4
Experiment
We compare our method, Wavelet-GS, with current state-of-the-
art scene reconstruction methods including 3DGS[14], 2DGS[10],
GOF[45], PGSR[3], Scaffold-GS[22] and Octree-GS[31]. Experiment
results are summarized in Fig. 5, Tab. 1 and Tab. 2. More results
could be found in supplementary material.
4.1
Experimental Settings
We evaluate our proposed method on four benchmark datasets
widely used in the 3D reconstruction community. Mip-NeRF360 [1],
a high-resolution dataset capturing 360-degree diverse indoor and
outdoor scenes; Waymo [37], a large-scale autonomous driving
dataset with real-world driving log. Tanks&Temples [17], a struc-
tured dataset comprising high-quality multi-view images of various
scenes; JHU-Drone [19], an aerial sub-dataset offering multi-view
daytime images of buildings from Johns Hopkins Homewood Cam-
pus Dataset. For our method, we set the voxel size to 0.001 for all
scenes and the number of neighbor nodes𝑘= 10 for all experiments.
We choose coif1 wavelet in the 3D and 2D wavelet decomposition.
The two loss weights 𝜆SSIM and 𝜆vol are set to 0.2 and 0.01 in our
experiments. For SH loss, we set 𝜆SH = 0.05.
4.2
Results Analysis
Our experimental results demonstrate the outstanding performance
of Wavelet-GS across a wide spectrum of evaluation metrics. Wavelet-
GS surpasses existing methodologies in both geometric consistency
and the preservation of fine-grained details in rendered outputs.
This performance is attributed to two core innovations: 3D wavelet
decomposition-based function and 2D wavelet-based relight mod-
ule. The 3D wavelet decomposition imposes strong initial con-
straints by efficiently capturing multiscale geometric details, laying
a robust foundation for reconstructing global structures and intri-
cate spatial features. Meanwhile, the 2D wavelet decomposition,
combined with the relight module, models environmental lighting
variations, ensuring spatial coherence and realistic color rendering
while preserving fine scene details.
The Waymo dataset presents unique challenges due to its sparse
input viewpoints, which make accurate 3D reconstruction difficult
for many existing methods. Wavelet-GS demonstrates an excep-
tional capacity to handle these constraints, achieving remarkable
fidelity and consistency even in scenarios characterized by sparse
coverage and intricate geometric structures.
As shown in Fig. 5, Wavelet-GS consistently outperforms state-
of-the-art methods in preserving fine-grained details, particularly
in regions with thin or complex geometries. Compared to other
techniques, our model exhibits superior spatial coherence and con-
tinuity in the reconstructed scenes. Furthermore, the visual results
highlight that Wavelet-GS generates high-fidelity reconstructions
with minimal artifacts, even under data-scarce conditions. Com-
peting approaches, in contrast, often fail to retain fine structural
details or produce inconsistent outputs, especially in challenging
areas.
On the Mip-NeRF360, JHU-Drone and Tanks&Temples dataset
that involve dense, panoramic inputs with complex lighting vari-
ations, Wavelet-GS showcases its ability to handle scenes with
intricate structure. The results illustrate that Wavelet-GS excels
in reconstructing high-quality scenes with intricate lighting and
geometric details. It outperforms other methods by producing visu-
ally coherent reconstructions that faithfully capture fine structural
variations. Moreover, the integration of wavelet decomposition
and relight significantly reduces artifacts in highly illuminated
or shadowed regions, where competing models often introduce
inconsistencies or fail to capture critical details.
Across all four datasets, Wavelet-GS achieves significant ad-
vancements in structural consistency and reconstruction fidelity.
By integrating the 3D and 2D wavelet decomposition function, our
approach ensures that reconstructed 3D points are geometrically
constrained and spatially coherent, even under diverse and chal-
lenging conditions. This seamless integration results in a robust
and reliable framework capable of producing accurate and visually
consistent 3D reconstructions.
4.3
Ablations
We evaluated the effectiveness of 3D and 2D wavelet function,
individual training strategy and LL-W, through ablation studies.
As shown in Tab. 2, the results confirm that all components are
essential for the enhanced performance observed in our model.
We also evaluated the influency of different wavelet families. The
ablation results are shown in Tab. 3. This ablation study shows the
choice of wavelet family can influence the reconstruction results.
4.3.1
Effect of 3D Wavelet Decomposition. The 3D Wavelet Decom-
position function is a cornerstone of our approach. Removing this
module significantly reduces the model’s ability to capture intricate
details and maintain structural consistency, particularly in high-
frequency regions like sharp edges or textured surfaces. The 3D
Wavelet Decomposition function ensures a strong initialization, en-
abling the model to focus on both coarse structures and fine details.

<!-- page 7 -->
Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Quantitative comparison to previous methods on real-world datasets. We have tested state-of-the-art 3DGS based
functions in four widely-used 3D dataset. The metric values are the average among different scenes of the dataset. More detail
results can be found in supplementary material.
Dataset
Waymo
JHU-Drone
Tanks&Temples
Mip-NeRF360
Method
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
3DGS [14]
0.840
27.19
0.313
0.882
29.60
0.116
0.850
23.88
0.171
0.870
28.69
0.182
GOF [45]
0.765
22.61
0.396
0.888
29.69
0.101
0.854
23.60
0.167
0.874
28.74
0.177
PGSR [3]
0.767
23.45
0.438
0.882
29.28
0.105
0.853
23.16
0.194
0.876
28.56
0.172
2DGS [10]
0.801
24.83
0.373
0.862
28.85
0.143
0.827
23.11
0.214
0.863
28.53
0.201
Scaffold-GS [22]
0.843
27.51
0.310
0.883
29.70
0.111
0.849
23.99
0.174
0.870
29.35
0.188
Octree-GS [31]
0.828
26.82
0.321
0.888
29.84
0.100
0.863
24.54
0.153
0.867
29.11
0.188
Ours
0.853
28.34
0.274
0.892
29.92
0.082
0.863
24.40
0.124
0.870
29.68
0.170
Figure 5: Comparison Results. Visual differences are highlighted with red insets for better clarity. Our approach consistently
outperforms Scaffold-GS [22] and Octree-GS [31] on Waymo dataset, demonstrating clear advantages in challenging scenarios
such as thin geometries and fine-scale details. Best viewed in color.
This improves point estimation accuracy and enhances robustness
in handling complex scenes with diverse spatial characteristics.
Furthermore, the wavelet decomposition framework enhances the
model’s ability to reconstruct fine-grained details in geometrically
intricate areas. For example, in scenarios with densely textured or
highly irregular surfaces, the module ensures that high-frequency
details are retained during the reconstruction process, leading to
more detailed and visually coherent results.
4.3.2
Effect of 2D Wavelet Decomposition. The 2D wavelet-based
relight module are critical for improving reconstruction quality,
particularly under varying lighting and surface reflectance. When
the 2D wavelet decomposition and relight module are removed
from the framework, a noticeable degradation in reconstruction
fidelity is observed. The relight Module dynamically normalizes
illumination across viewpoints and incorporates reflectance-aware
adjustments, reducing inconsistencies caused by lighting variations.
This ensures geometrically accurate and visually coherent recon-
structions. Additionally, it mitigates artifacts from uneven lighting,
preserving fine-grained details, especially in intricate regions. The
module contributes to the preservation of fine-grained details by
mitigating artifacts introduced by uneven lighting, particularly in
geometrically intricate regions.

<!-- page 8 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, and Hao Wang*
Table 2: Quantitative ablation comparison on real-world
datasets. The ablation experiment results on Waymo dataset
prove all components are essential for the enhanced perfor-
mance observed in our model.
Method
SSIM↑
PSNR↑
LPIPS↓
w/o individual strategy
0.840
27.22
0.312
w/o 2D&3D Wavelet
0.841
27.48
0.322
w/o LL-W
0.844
27.88
0.303
w/o 3D Wavelet
0.843
27.58
0.310
w/o 2D Wavelet
0.845
27.91
0.289
Ours
0.853
28.34
0.274
Table 3: Quantitative ablation comparison on real-world
datasets. The ablation experiment results on Waymo dataset
compare the effects of different wavelet families.
Method
SSIM↑
PSNR↑
LPIPS↓
haar
0.847
28.21
0.282
db8
0.849
28.15
0.284
sym16
0.846
28.24
0.281
coif1 (Ours)
0.853
28.34
0.274
4.3.3
Effect of Individual Strategy and LL-W. The individual train-
ing strategies and LL-W are essential for balancing structural preci-
sion and detail preservation. When these customized strategies are
replaced with the original 3D Gaussian Splatting (3DGS) optimiza-
tion, the model’s ability to handle multi-frequency components
deteriorates, leading to either oversmoothing (due to insufficient
high-frequency modeling) or fragmented artifacts (caused by un-
stable optimization in high-frequency regions). This highlights the
necessity of frequency-aware training mechanisms for robust re-
construction.
The loss function LL-W ensures detail feature preservation by pe-
nalizing discrepancies across spatial and frequency domains. With-
out it, detail fidelity, especially in high-frequency regions, degrades
significantly, as the model loses critical guidance for fine textures
and edges. By imposing penalties on discrepancies that arise across
both spatial and frequency domains, this loss function provides
essential guidance for the model during training.
4.3.4
Effect of different wavelet families. A critical aspect of wavelet-
based analysis is the selection of an appropriate mother wavelet, as
different wavelet families possess distinct characteristics in terms
of symmetry, regularity, support length, and number of vanishing
moments. These properties can significantly impact the efficiency
of signal representation and feature extraction for a given task. To
evaluate the sensitivity of our proposed method’s performance to
this choice, we conducted an ablation study comparing four repre-
sentative wavelet families: Haar (haar), Daubechies 8 (db8), Symlet
16 (sym16), and Coiflet 1 (coif1). The experimental setup was kept
consistent with our main configuration, varying only the wavelet
basis used for the decomposition stage.
The Coiflet 1 (coif1) wavelet consistently yielded the superior
results, achieving the highest performance metrics among the eval-
uated candidates. Coiflets are known for their near symmetry and
relatively high number of vanishing moments for both the wavelet
(𝜓) and scaling (𝜙) functions relative to their support size. This
balance appears particularly advantageous for capturing the rel-
evant morphological features within our data, leading to a more
discriminative feature representation.
This comparative analysis underscores that the choice of wavelet
family can influence the efficacy of wavelet-based methods. For
our specific application and dataset, the unique properties of the
coif1 wavelet provided the most effective basis for signal analysis,
justifying its selection in our final proposed model.
5
Discussion
In this work, we leverage 3D and 2D wavelet decomposition-based
3DGS to improve the robustness and accuracy of 3D scene recon-
struction. The Wavelet Decomposition module provides a strong
initialization by isolating features across scales, allowing the model
to focus on both coarse geometric structures and fine local de-
tails. This multiscale representation not only improves the accu-
racy of point estimation but also enhances the model’s robustness
in handling complex scenes with diverse spatial characteristics.
However, there are limitations to our approach. Although wavelet
decomposition-based 3DGS effectively captures structural informa-
tion, it requires more computational cost and memory to restore the
information. Additionally, while the relight module significantly
improves detail in high-frequency regions, it may introduce com-
putational overhead, particularly when processing complex scenes
with dense textures.
6
Conclusions
In this paper, we presented a wavelet decomposition-based frame-
work for 3D scene reconstruction, which separates point clouds into
high and low frequency components for individual training opti-
mization. The low-frequency component captures global structural
outlines, while the high-frequency component restores intricate
details, ensuring improved fidelity. We further propose a gradient
and opacity based training strategy to enhance the structural rep-
resentation of the low-frequency component. Additionally, the 2D
wavelet-based relight module in the high-frequency component
models lighting effects, enhancing photorealism. Experiments on
challenging datasets demonstrate the superiority of our framework
over state-of-the-art methods in structural accuracy, detail fidelity,
and rendering quality. This work advances robust 3D reconstruc-
tion and opens pathways for efficient and realistic scene modeling
in complex environments.
Acknowledgment
This research is supported by the National Natural Science Founda-
tion of China (No. 62406267), Guangzhou-HKUST(GZ) Joint Fund-
ing Program (Grant No.2025A03J3956 & Grant No.2023A03J0008),
the Guangzhou Municipal Science and Technology Project (No.
2025A04J4070), and the Guangzhou Municipal Education Project
(No. 2024312122).

<!-- page 9 -->
Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter
Hedman. 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
5470–5479.
[2] Debra Charnley and Rod Blissett. 1989. Surface reconstruction from outdoor
image sequences. Image and Vision Computing 7, 1 (1989), 10–16.
[3] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan
Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. 2024. PGSR: Planar-based
Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction. arXiv
preprint arXiv:2406.06521 (2024).
[4] Guikun Chen and Wenguan Wang. 2024. A survey on 3d gaussian splatting.
arXiv preprint arXiv:2401.03890 (2024).
[5] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. 2022. Depth-
supervised nerf: Fewer views and faster training for free. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12882–12891.
[6] Kyle Gao, Yina Gao, Hongjie He, Dening Lu, Linlin Xu, and Jonathan Li. 2022.
Nerf: Neural radiance field in 3d vision, a comprehensive review. arXiv preprint
arXiv:2210.00379 (2022).
[7] Amara Graps. 1995. An introduction to wavelets. IEEE computational science and
engineering 2, 2 (1995), 50–61.
[8] Robin Green. 2003. Spherical harmonic lighting: The gritty details. In Archives of
the game developers conference, Vol. 56. 4.
[9] Martin Habbecke and Leif Kobbelt. 2007. A surface-growing approach to multi-
view stereo reconstruction. In 2007 IEEE Conference on Computer Vision and
Pattern Recognition. IEEE, 1–8.
[10] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao.
2024. 2d gaussian splatting for geometrically accurate radiance fields. In ACM
SIGGRAPH 2024 conference papers. 1–11.
[11] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen
Li, Henry Lau, Feng Gao, Yin Yang, et al. 2024. Vr-gs: A physical dynamics-aware
interactive gaussian splatting system in virtual reality. In ACM SIGGRAPH 2024
Conference Papers. 1–1.
[12] Hailin Jin, Stefano Soatto, and Anthony J Yezzi. 2005. Multi-view stereo re-
construction of dense shape and complex appearance. International Journal of
Computer Vision 63 (2005), 175–189.
[13] Joanna Kaleta, Kacper Kania, Tomasz Trzcinski, and Marek Kowalski. 2024. Lu-
miGauss: High-Fidelity Outdoor Relighting with 2D Gaussian Splatting. arXiv
preprint arXiv:2408.04474 (2024).
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans.
Graph. 42, 4 (2023), 139–1.
[15] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer,
Alexandre Lanvin, and George Drettakis. 2024. A hierarchical 3d gaussian repre-
sentation for real-time rendering of very large datasets. ACM Transactions on
Graphics (TOG) 43, 4 (2024), 1–15.
[16] Hansung Kim, Jean-Yves Guillemaut, Takeshi Takai, Muhammad Sarim, and
Adrian Hilton. 2012. Outdoor dynamic 3-D scene reconstruction. IEEE Transac-
tions on Circuits and Systems for Video Technology 22, 11 (2012), 1611–1622.
[17] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and
temples: Benchmarking large-scale scene reconstruction. ACM Transactions on
Graphics (ToG) 36, 4 (2017), 1–13.
[18] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt.
2020. Neural sparse voxel fields. Advances in Neural Information Processing
Systems 33 (2020), 15651–15663.
[19] Xijun Liu, Yifan Zhou, Yuxiang Guo, Rama Chellappa, and Cheng Peng. 2024. An
Immersive Multi-Elevation Multi-Seasonal Dataset for 3D Reconstruction and
Visualization. arXiv preprint arXiv:2412.14418 (2024).
[20] Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang
Zhang. 2025. Citygaussian: Real-time high-quality large-scale scene rendering
with gaussians. In European Conference on Computer Vision. Springer, 265–282.
[21] Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser
Sheikh, and Jason Saragih. 2021. Mixture of volumetric primitives for efficient
neural rendering. ACM Transactions on Graphics (ToG) 40, 4 (2021), 1–13.
[22] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo
Dai. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
20654–20664.
[23] He Lyu, Ningyu Sha, Shuyang Qin, Ming Yan, Yuying Xie, and Rongrong Wang.
2019. Advances in neural information processing systems. Advances in neural
information processing systems 32 (2019).
[24] Zhiliang Ma and Shilong Liu. 2018. A review of 3D reconstruction techniques in
civil engineering and their applications. Advanced Engineering Informatics 37
(2018), 163–174.
[25] Stephane Mallat. 1999. A wavelet tour of signal processing.
[26] Stephane G Mallat. 1989. A theory for multiresolution signal decomposition:
the wavelet representation. IEEE transactions on pattern analysis and machine
intelligence 11, 7 (1989), 674–693.
[27] Yves Meyer. 1992. Wavelets and operators: volume 1. Number 37. Cambridge
university press.
[28] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi
Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance
fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
[29] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon
Giebenhain, and Matthias Nießner. 2024. Gaussianavatars: Photorealistic head
avatars with rigged 3d gaussians. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 20299–20309.
[30] Ravi Ramamoorthi and Pat Hanrahan. 2001. An efficient representation for
irradiance environment maps. In Proceedings of the 28th annual conference on
Computer graphics and interactive techniques. 497–500.
[31] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai.
2024. Octree-gs: Towards consistent real-time rendering with lod-structured 3d
gaussians. arXiv preprint arXiv:2403.17898 (2024).
[32] Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan Li, and Giljoo Nam.
2024. Relightable gaussian codec avatars. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition. 130–141.
[33] Thomas Schöps, Torsten Sattler, Christian Häne, and Marc Pollefeys. 2017. Large-
scale outdoor 3D reconstruction on a mobile device. Computer Vision and Image
Understanding 157 (2017), 151–166.
[34] Steven M Seitz, Brian Curless, James Diebel, Daniel Scharstein, and Richard
Szeliski. 2006. A comparison and evaluation of multi-view stereo reconstruction
algorithms. In 2006 IEEE computer society conference on computer vision and
pattern recognition (CVPR’06), Vol. 1. IEEE, 519–528.
[35] Gaochao Song, Chong Cheng, and Hao Wang. 2024. GVKF: Gaussian Voxel
Kernel Functions for Highly Efficient Surface Reconstruction in Open Scenes.
arXiv preprint arXiv:2411.01853 (2024).
[36] Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022. Direct voxel grid optimiza-
tion: Super-fast convergence for radiance fields reconstruction. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition. 5459–5469.
[37] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai
Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al.
2020. Scalability in perception for autonomous driving: Waymo open dataset. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
2446–2454.
[38] Jizeng Wang, Xiaojing Liu, and Youhe Zhou. 2024. Application of wavelet methods
in computational physics. Annalen der Physik 536, 5 (2024), 2300461.
[39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image
quality assessment: from error visibility to structural similarity. IEEE transactions
on image processing 13, 4 (2004), 600–612.
[40] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-Pei Cao, Ling-Qi Yan, and
Lin Gao. 2024. Recent advances in 3d gaussian splatting. Computational Visual
Media 10, 4 (2024), 613–642.
[41] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chen-
fanfu Jiang. 2024. Physgaussian: Physics-integrated 3d gaussians for generative
dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 4389–4398.
[42] Jin-Chao Xu and Wei-Chang Shann. 1992. Galerkin-wavelet methods for two-
point boundary value problems. Numer. Math. 63, 1 (1992), 123–144.
[43] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan,
Xianpeng Lang, Xiaowei Zhou, and Sida Peng. 2024. Street gaussians: Model-
ing dynamic urban scenes with gaussian splatting. In European Conference on
Computer Vision. 156–173.
[44] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024.
Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 19447–19456.
[45] Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024. Gaussian opacity fields:
Efficient and compact surface reconstruction in unbounded scenes. arXiv preprint
arXiv:2404.10772 (2024).
[46] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang,
Liqiang Nie, and Yebin Liu. 2024. Gps-gaussian: Generalizable pixel-wise 3d
gaussian splatting for real-time human novel view synthesis. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19680–19690.
[47] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-
Hsuan Yang. 2024. Drivinggaussian: Composite gaussian splatting for surround-
ing dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition. 21634–21643.
[48] Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Justus
Thies, and Javier Romero. 2023. Drivable 3d gaussian avatars. arXiv preprint
arXiv:2311.08581 (2023).
[49] Michael Zollhöfer, Patrick Stotko, Andreas Görlitz, et al. 2018. State of the art
on 3D reconstruction with RGB-D cameras. In Computer graphics forum, Vol. 37.
Wiley Online Library, 625–652.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
