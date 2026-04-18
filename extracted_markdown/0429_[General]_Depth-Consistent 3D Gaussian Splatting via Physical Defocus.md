<!-- page 1 -->
Depth-Consistent 3D Gaussian Splatting via Physical Defocus
Modeling and Multi-View Geometric Supervision
Yu Denga, Baozhu Zhaoa, Junyan Sua, Xiaohan Zhanga and Qi Liua,∗
aDepartment of Future Technology, South China University of Technology, Guangzhou, 511400, China
A R T I C L E I N F O
Keywords:
3D scene reconstruction
Depth-of-field
Differentiable rendering
Novel view synthesis
A B S T R A C T
Three-dimensional reconstruction in scenes with extreme depth variations remains challenging due
to inconsistent supervisory signals between near-field and far-field regions. Existing methods fail to
simultaneously address inaccurate depth estimation in distant areas and structural degradation in
close-range regions. This paper proposes a novel computational framework that integrates depth-
of-field supervision and multi-view consistency supervision to advance 3D Gaussian Splatting. Our
approach comprises two core components: (1) Depth-of-field Supervision employs a scale-recovered
monocular depth estimator (e.g., Metric3D) to generate depth priors, leverages defocus convolution
to synthesize physically accurate defocused images, and enforces geometric consistency through a
novel depth-of-field loss, thereby enhancing depth fidelity in both far-field and near-field regions;
(2) Multi-View Consistency Supervision employing LoFTR-based semi-dense feature matching to
minimize cross-view geometric errors and enforce depth consistency via least squares optimization
of reliable matched points. By unifying defocus physics with multi-view geometric constraints, our
method achieves superior depth fidelity, demonstrating a 0.8 dB PSNR improvement over the state-
of-the-art method on the Waymo Open Dataset. This framework bridges physical imaging principles
and learning-based depth regularization, offering a scalable solution for complex depth stratification
in urban environments.
1. Introduction
Three-dimensional scene reconstruction from multi-
view images remains a cornerstone capability for applica-
tions ranging from autonomous driving to immersive virtual
reality. While Neural Radiance Fields (NeRF) (Milden-
hall, Srinivasan, Tancik, Barron, Ramamoorthi and Ng,
2021) revolutionized photorealistic novel view synthesis,
subsequent advances in 3D Gaussian Splatting (3DGS)
(Kerbl, Kopanas, Leimkühler and Drettakis, 2023) have
achieved unprecedented real-time rendering speeds through
differentiable Gaussian primitives. However, these methods
face critical limitations when reconstructing scenes with
substantial depth variations, as distant structures often ex-
hibit positional inaccuracies due to insufficient supervision
signals and overfitting to training viewpoints.
Current approaches for large-scale scene reconstruction
rely predominantly on multi-view stereo (MVS) techniques
(Furukawa, Hernández et al., 2015) or volumetric neural
representations (Barron, Mildenhall, Verbin, Srinivasan and
Hedman, 2022). The former establishes geometric consis-
tency through hand-crafted features (Xu and Tao, 2019),
while the latter optimizes implicit fields through photo-
metric loss (Wang, Liu, Chen, Liu, Liu, Komura, Theobalt
and Wang, 2023). Recent extensions such as GaussianPro
(Cheng, Long, Yang, Yao, Yin, Ma, Wang and Chen, 2024a)
attempt to mitigate extreme distant interference through sky
∗Corresponding author
202311093429@mail.scut.edu.cn (Y. Deng);
202320163293@mail.scut.edu.cn (B. Zhao); ft_su.junyan@mail.scut.edu.cn
(J. Su); ftxiaohanzhang@mail.scut.edu.cn (X. Zhang); drliuqi@scut.edu.cn
(Q. Liu)
https://drliuqi.github.io/ (Q. Liu)
ORCID(s): 0000-0001-5378-6404 (Q. Liu)
segmentation masks, yet introduce new artifacts from imper-
fect matting and sparse depth supervision. This limitation
stems from a fundamental challenge: when objects appear
at varying distances across frames, conventional supervision
struggles to resolve scale ambiguities, particularly for distant
regions receiving insufficient pixel-level gradients.
Our work addresses these limitations through two syn-
ergistic innovations. First, we leverage depth-of-field effects
as implicit geometric supervision. By modeling the physical
correlation between defocus blur and scene depth via adap-
tive kernel convolution, we derive gradient signals that guide
Gaussians towards their geometrically consistent positions.
Second, we introduce a hybrid depth estimation frame-
work that integrates multi-view feature matching (LoFTR)
(Wang, He, Peng, Tan and Zhou, 2024c) with monocular
depth completion (Metric3D) (Hu, Yin, Zhang, Cai, Long,
Chen, Wang, Yu, Shen and Shen, 2024a), resolving scale
ambiguities while preserving structural details. This dual
strategy effectively constrains Gaussian distributions across
varying depth layers, particularly benefiting distant regions
reconstruction where traditional methods fail.
The technical contributions of this work are threefold:
• A physics-aware defocus convolution model that trans-
lates optical principles into geometric constraints,
using adaptive kernel designs (Gaussian, Polygonal,
or SmoothStep) to adapt to different types of cameras
and dynamic focus optimization to enhance depth
consistency.
• A multiscale depth alignment framework combin-
ing global monocular depth recovery with local grid-
based correction, achieving view-consistent depth es-
timation without manual masking.
: Preprint submitted to Elsevier
Page 1 of 19
arXiv:2511.10316v1  [cs.CV]  13 Nov 2025

<!-- page 2 -->
• A gradient-aware density control mechanism that pri-
oritizes structurally critical regions through depth-
gradient statistics.
Extensive validation on urban (Waymo) and unbounded
(Mip-NeRF 360) scenes demonstrates the efficacy of our
approach. Quantitative results show that our model achieves
35.17 PSNR on Waymo, outperforming SOTA methods.
Qualitative analyzes reveal significant improvements in near
region structure recovery, particularly for vehicles and build-
ings. These advancements establish new state-of-the-art per-
formance for depth-aware scene reconstruction while pre-
serving the computational efficiency central to 3DGS frame-
works.
2. Related Work
2.1. Multi-View Stereo
MVS represents a fundamental computer vision task
that aims to reconstruct high-fidelity 3D models from a
collection of calibrated images. The existing MVS methods
can be broadly categorized into traditional geometry-based
approaches and contemporary learning-based approaches.
Traditional MVS approaches typically derive their cam-
era parameters predominantly from Structure-from-Motion
(SfM) methods (Snavely, Seitz and Szeliski, 2006; Schon-
berger and Frahm, 2016) or Simultaneous Localization and
Mapping (SLAM) frameworks (Mur-Artal, Montiel and
Tardos, 2015; Engel, Schöps and Cremers, 2014). Within
this paradigm, seminal methods proposed by Campbell,
Vogiatzis, Hernández and Cipolla (2008), Furukawa et al.
(2015), and Xu and Tao (2019) establish explicit pixel
correspondences through hand-crafted features and rigorous
geometric constraints.
Learning-based approaches (Kar, Häne and Malik, 2017;
Ji, Gall, Zheng, Liu and Fang, 2017; Zhou, Zhao, Wang,
Hao and Lei, 2023) have revolutionized MVS, initiated by
the pioneering end-to-end architecture introduced by Yao,
Luo, Li, Fang and Quan (2018). Contemporary methods
(Ma, Teed and Deng, 2022b; Feng, Yang, Guo and Li, 2023)
leverage learned representations for robust depth regression,
while advanced techniques such as cascade cost volumes
(Gu, Fan, Zhu, Dai, Tan and Tan, 2020) and feature matching
networks (Giang, Song and Jo, 2021) have substantially
enhanced both performance and computational efficiency.
2.2. Neural Radiance Field
A radiance field establishes a mapping from a 3D spatial
coordinate (𝑥, 𝑦, 𝑧) and viewing directions parameterized by
polar angle 𝜃and azimuthal angle 𝜙to a nonnegative radi-
ance value, characterizing the light-matter interaction within
the environment (Chen and Wang, 2024). NeRF (Milden-
hall et al., 2021) pioneered the paradigm of representing a
scene as an emissive volumetric function, implemented via a
position-encoded neural network that enables differentiable
rendering through volumetric quadrature.
Volumetric representations integrated with deep-learning
techniques and volumetric ray-marching were initially pro-
posed by Sitzmann, Thies, Heide, Nießner, Wetzstein and
Zollhofer (2019) and Henzler, Mitra and Ritschel (2019).
Significant advancements in this domain include Instant-
NGP (Müller, Evans, Schied and Keller, 2022), which em-
ploys multi-resolution hash grids, and Plenoxels (Fridovich-
Keil, Yu, Tancik, Chen, Recht and Kanazawa, 2022), which
utilizes sparse voxel grids for efficient optimization. Several
approaches enhance rendering efficiency through scene
reparameterization to generate more compact representa-
tions, notably Mip-NeRF 360 (Barron et al., 2022), Zip-
NeRF (Barron, Mildenhall, Verbin, Srinivasan and Hedman,
2023), and F2-NeRF (Wang et al., 2023).
Recent research has increasingly focused on addressing
defocus blur in neural rendering, with seminal works includ-
ing RawNeRF (Mildenhall, Hedman, Martin-Brualla, Srini-
vasan and Barron, 2022), AR-NeRF (Kaneko, 2022), and
NeRFocus (Wang, Yang, Hu and Zhang, 2022), all capable
of synthesizing depth-of-field effects. Subsequently, Deblur-
NeRF (Ma, Li, Liao, Zhang, Wang, Wang and Sander,
2022a) mitigates image blurriness resulting from defocus by
implementing a Deformable Sparse Kernel module, while
DP-NeRF (Lee, Lee, Shin and Lee, 2023) specifically ad-
dresses geometric and appearance consistency challenges
in defocused scenarios. Nevertheless, as noted by (Wang,
Chakravarthula and Chen, 2024b), these approaches con-
tinue to encounter substantial limitations regarding compu-
tational efficiency and real-time rendering capabilities.
2.3. 3D Gaussian Splatting
Novel view synthesis has evolved significantly with
NeRF (Mildenhall et al., 2021) setting a milestone for pho-
torealistic rendering. Building upon this foundation, 3DGS
(Kerbl et al., 2023) introduces a paradigm shift that models
scenes as a collection of 3D Gaussian primitives rendered
via differentiable rasterization, simultaneously achieving
high-quality reconstruction and real-time rendering capa-
bilities. This approach extends traditional splatting-based
rasterization (Zwicker, Pfister, Van Baar and Gross, 2002)
by optimizing Gaussian primitives with explicit geometry
and appearance attributes (Cheng et al., 2024a; Li, Shi, Cao,
Ni, Zhang, Zhang and Van Gool, 2024).
Recent advances in 3DGS research have predominantly
focused on efficiency and quality enhancements. Innovative
methods have been proposed to optimize Gaussian contri-
butions through scale-based evaluation strategies (Lee, Rho,
Sun, Ko and Park, 2024) and sophisticated visibility as-
sessment techniques (Fan, Wang, Wen, Zhu, Xu and Wang,
2023). Concurrent developments have yielded substantial
improvements in rendering fidelity (Yu, Chen, Huang, Sat-
tler and Geiger, 2024; Blanc, Deschaud and Paljic, 2024;
Huang, Yu, Chen, Geiger and Gao, 2024), computational
efficiency (Girish, Gupta and Shrivastava, 2024), and the
capacity to handle large-scale scenes with complex geome-
try (Kerbl, Meuleman, Kopanas, Wimmer, Lanvin and Dret-
takis, 2024; Liu, Luo, Fan, Wang, Peng and Zhang, 2024b).
: Preprint submitted to Elsevier
Page 2 of 19

<!-- page 3 -->
Figure 1: A schematic illustrating the principle of depth of field
blur. When a scene point M at a distance d does not lie on the
focus plane (at distance 𝑑𝑓), it creates a blurred spot on the
image plane known as the circle of confusion (with a diameter
of 𝐷𝑐𝑜𝑐), causing the image to be out of focus. f represents the
focal length of the lens.
Despite these significant advancements, fundamental
limitations persist in the 3DGS representation framework.
The inherent absence of true volumetric density fields man-
ifests as view-dependent consistency issues and rendering
artifacts (Radl, Steiner, Parger, Weinrauch, Kerbl and Stein-
berger, 2024; Mai, Hedman, Kopanas, Verbin, Futschik, Xu,
Kuester, Barron and Zhang, 2024). Furthermore, the conven-
tional pinhole camera model employed in standard 3DGS
implementations inherently restricts its application domain
to All-in-Focus (AiF) scenarios (Wang et al., 2024b).
Recent research has addressed these fundamental chal-
lenges through innovative depth-of-field rendering approaches.
DOF-GS (Wang et al., 2024b) introduces a finite aperture
camera model coupled with explicit, differentiable defo-
cus rendering guided by the Circle-of-Confusion (CoC),
thereby enabling both adjustable depth-of-field effects and
the generation of AiF images from defocused training data.
Similarly, Cinematic Gaussians (Wang, Wolski, Kerbl, Ser-
rano, Bemana, Seidel, Myszkowski and Leimkühler, 2024a)
leverages multi-view LDR images with varying exposure
times, apertures, and focus distances to reconstruct high-
dynamic-range (HDR) radiance fields, incorporating analyt-
ical convolutions of Gaussians based on a thin-lens camera
model. Unlike these prior works that primarily focus on ei-
ther depth-of-field effects rendering or HDR reconstruction,
our approach uniquely leverages depth-of-field information
as an additional structural supervision signal to achieve more
geometrically accurate 3D reconstruction.
3. Method
Our computational framework integrates dual supervi-
sory paradigms for geometrically consistent 3D Gaussian
Splatting, as depicted in Figure 2. The architecture oper-
ates on multi-view inputs through dual-branch processing,
establishing metric-scale depth priors while enforcing cross-
view consistency constraints. Guided by this computational
paradigm, we first introduce our depth-aware defocus mod-
eling that formulates physical optics principles as differen-
tiable geometric constraints. Subsequent sections systemati-
cally elaborate our hybrid depth estimation framework com-
prising global scale calibration and local depth refinement
modules, culminating in multi-view geometric consistency
enforcement.
3.1. Defocus Convolution
Accurate modeling of defocus effects constitutes an es-
sential component in establishing comprehensive geomet-
ric and radiometric supervision signals (Cui and Knoll,
2024) for 3D Gaussian Splatting-based scene reconstruction.
This section presents a physics-driven defocus convolution
framework that emulates physical imaging characteristics to
achieve high-fidelity depth-of-field reconstruction through
optical principle formulation.
3.1.1. Physical Imaging Model
As shown in Figure 1, in optical imaging systems,
scene points at distinct depth layers generate varying Circle
of Confusion (CoC) dimensions through lens propagation.
Considering a 3D scene point with object distance 𝑑, focal
length 𝑓, and focus distance 𝑑𝑓, the CoC diameter in optical
space is derived as:
𝐷(coc) =
𝑓2|𝑑−𝑑𝑓|
𝐹⋅𝑑⋅(𝑑𝑓−𝑓)
(1)
where 𝐹represents the f-number (defined as 𝑓∕𝐴, with
𝐴denoting the physical aperture diameter). For digital imag-
ing applications, we convert the optical CoC diameter to
pixel space through sensor-image scaling:
𝐷(pixel) = 𝐷(coc) ⋅𝑤𝑖
𝑤𝑠
(2)
where 𝐷(pixel) corresponds to the effective defocus diam-
eter in digital coordinates, 𝑤𝑖indicates the image resolution
width (pixels), and 𝑤𝑠specifies the sensor’s physical width
(mm).
3.1.2. Adaptive Kernel Design
Our framework incorporates three distinct convolution
kernels to address diverse defocus characteristics found in
optical systems. This multi-kernel approach provides the
flexibility to balance physical realism with computational
efficiency.
Gaussian Blur Kernel For baseline defocus simulation,
we employ a standard Gaussian kernel. Its softness provides
a natural-looking blur. The kernel is defined as:
𝐺(𝑥, 𝑦) =
1
2𝜋𝜎2 exp
(
−𝑥2 + 𝑦2
2𝜎2
)
(3)
where 𝜎relates to the circle of confusion (CoC) radius
𝑅through the formulation:
: Preprint submitted to Elsevier
Page 3 of 19

<!-- page 4 -->
Figure 2: Our framework consists of two core technical components: (a) Depth-of-Field Supervision (Blue Flow) addressing
inaccuracies in distant scenes and difficulties in recovering structures in near-field scenes. The pipeline takes multi-view images as
input, obtains scale-ambiguous depth predictions through a monocular depth estimator (e.g., Metric3D), and calculates true depth
maps via a multi-view depth scale recovery algorithm. Defocus convolution is then utilized to generate defocused images from both
rendered and ground truth images, with the final dof loss between these defocused images supervising the 3DGS training. (b)
Multi-View Consistency Supervision (Orange Flow) resolving cross-view geometric alignment issues. Initially, semi-dense feature
matching is performed across multi-view images using LoFTR, minimizing the error geo between 3D points corresponding to
matched pixels to enhance cross-view geometric consistency. Simultaneously, a depth consistency loss depth employs local depth
maps recovered through least squares optimization from accurately matched points with reliable depth information to optimize
the depth rendered by 3DGS.
𝜎𝐺= 𝐷(pixel)
𝑘𝑠
(4)
Here, 𝑘𝑠represents a normalization coefficient (default:
20) that scales the physical CoC diameter to kernel space.
To ensure energy conservation, we enforce unitary integral
constraint via kernel normalization:
𝐾norm =
𝐺
∑
𝑥,𝑦𝐺(𝑥, 𝑦)
(5)
SmoothStep Blur Kernel To better preserve sharp
edges in regions with high depth discontinuity, a known
limitation of Gaussian blur, we implement a hyperbolic
tangent-based kernel. Its S-shaped transition profile offers
a superior trade-off between blurring and edge preservation.
The formulation is:
𝐾(𝑥, 𝑦) = 0.5+0.5 tanh (0.25(𝑟2 −𝑥2 −𝑦2) + 0.5) (6)
where r denotes the kernel radius, and (x,y) represent
the coordinates of each position within the kernel. This
formulation achieves controlled edge preservation through
its S-shaped transition profile, particularly effective for depth
discontinuity regions.
Polygonal Blur Kernel To achieve the highest degree
of physical realism, especially for simulating the character-
istic ’bokeh’ from a lens’s aperture blades, we introduce a
parametric polygonal kernel. As shown in Fig. 3, this model
explicitly encodes the aperture geometry. The formulation is
as follows:
𝛼Poly
𝑖
(𝑝) = 𝑜𝑖⋅𝛽𝑖⋅𝐾(𝑝)
(7)
where 𝑁(default: 8) represents the number of aperture
blades, 𝛽𝑖= 1∕∑
𝑝∈Ω 𝐾(𝑝) ensures normalization, and 𝐾(𝑝)
integrates radial attenuation 𝑊(𝑟) with geometric contain-
ment 𝐻(𝑝):
𝐾(𝑝) = 𝐻(𝑝) ⋅𝑊(𝑟(𝑝))
(8)
where 𝑟(𝑝) =
√
𝑥2 + 𝑦2 represents the Euclidean dis-
tance from point p to the center. The radial weight function
employs cosine attenuation:
: Preprint submitted to Elsevier
Page 4 of 19

<!-- page 5 -->
𝑊(𝑟) = cos
(𝜋
2 ⋅𝑟
𝑅
)
⋅𝟏[𝑟≤𝑅]
(9)
where 𝑅denotes the predefined kernel radius, 𝟏[𝑟≤
𝑅] is the indicator function. The vertices of the polygon
are given by 𝑣𝑖=
(
𝑅cos
(
2𝜋𝑖
𝑁
)
, 𝑅sin
(
2𝜋𝑖
𝑁
))
for 𝑖=
1, 2, … , 𝑁, where 𝑣𝑁+1 = 𝑣1 ensures the polygon closure.
The cross-product function 𝐶(𝑝, 𝑣𝑖, 𝑣𝑖+1), which deter-
mines the relative position of point 𝑝with respect to the edge
formed by vertices 𝑣𝑖and 𝑣𝑖+1:
𝐶(𝑝, 𝑣𝑖, 𝑣𝑖+1) = (𝑣𝑖+1,𝑥−𝑣𝑖,𝑥)(𝑝𝑦−𝑣𝑖,𝑦)
−(𝑣𝑖+1,𝑦−𝑣𝑖,𝑦)(𝑝𝑥−𝑣𝑖,𝑥).
(10)
Using this cross-product criterion, the indicator function
𝐻(𝑝), which determines whether a point 𝑝resides inside the
𝑁-sided polygon based on the cross-product criterion:
𝐻(𝑝) =
{
1,
if ⋀𝑁
𝑖=1 𝐶(𝑝, 𝑣𝑖, 𝑣𝑖+1) < 0
0,
otherwise
(11)
Figure 3: Illustration of polygonal aperture mechanisms in a
camera lens: (a) octagon aperture blades and (b) dodecagon
aperture blades.
This parametric modeling accurately reproduces the op-
tical characteristics observed in real-world Polygonal aper-
tures, as evidenced by the qualitative comparison in Fig-
ure 3(b).
The distinct visual effects of these three kernels are
compared in Figure 4, guiding the choice of kernel for
different application requirements.
3.1.3. Dynamic Focus Optimization
We optimize focus distance 𝑑𝑓using depth distribution
statistics {𝑑1∕3, 𝑑1∕2, 𝑑2∕3, 𝜇𝑑} (terciles and mean depth).
The optimal focus distance minimizes:
𝑑∗
𝑓= arg min
𝑑𝑓
∑
𝑝∈Ω
𝑤(𝑝)‖𝑑(𝑝) −𝑑𝑓‖2
(12)
(a) No Blur Kernel
(b) Gaussian Blur Kernel
(c) SmoothStep Blur Kernel
(d) Polygonal Blur Kernel
Figure 4: Comparative analysis of defocus convolution tech-
niques: (a) Original (no blur) provides baseline sharpness; (b)
implementing radially symmetric blur via bell-shaped inten-
sity profiles to simulate natural defocus; (c) preserving edge
structures through S-curve transitions using hyperbolic tangent
functions; (d) emulating optical apertures with geometric
containment and radial attenuation for realistic bokeh effects.
3.1.4. Defocus Synthesis
Given an input image 𝐼, the final defocused image 𝐼out
is synthesized as follows:
𝐼out(𝑝) =
∑
𝑞∈(𝑝)
𝐼(𝑞)𝐾(𝑅(𝑑𝑝)) (𝑝−𝑞)
(13)
where (𝑝) denotes the neighborhood of pixel 𝑝, 𝐾(𝑅)
represents the convolution kernel associated with the CoC
size 𝑅, which is determined by the depth 𝑑𝑝of the scene
point corresponding to pixel 𝑝.
To enhance computational efficiency while maintaining
physical accuracy, we implement a separable convolution
approach. Specifically, we decompose the 2D convolution
into two sequential 1D convolutions along the horizontal and
vertical directions:
𝐼out = (𝐼∗𝑘𝑥
) ∗𝑘𝑦
(14)
where 𝑘𝑥and 𝑘𝑦are 1D kernel functions in the hori-
zontal and vertical directions, respectively. This separable
implementation reduces the computational complexity from
(𝑛2) to (2𝑛) for an 𝑛× 𝑛kernel, significantly accelerating
the processing while preserving the physical fidelity of the
defocus effects.
: Preprint submitted to Elsevier
Page 5 of 19

<!-- page 6 -->
3.1.5. Depth-of-Field Loss
Our loss design addresses two critical and distinct as-
pects of the reconstruction: ensuring photometric accuracy
in sharp, in-focus regions, and enforcing physical plausibil-
ity of blur in out-of-focus regions. This dual supervision
is key to achieving both geometric accuracy and optical
realism.
Supervision for In-Focus Regions. To optimize sharp
regions, we employ a standard reconstruction loss, rgb,
which combines a pixel-wise L1 loss and a structural sim-
ilarity (SSIM) term:
rgb = (1 −𝜆dssim)(rgb)
L1
+ 𝜆dssim(rgb)
SSIM
(15)
The constituent losses are defined as:
(rgb)
L1
=
1
|Ω|
∑
𝑝∈Ω
‖𝐼rend(𝑝) −𝐼gt(𝑝)‖1
(16)
(rgb)
SSIM = 1 −
(2𝜇rend𝜇gt + 𝐶1)(2𝜎rend𝜎gt + 𝐶2)
(𝜇2
rend + 𝜇2
gt + 𝐶1)(𝜎2
rend + 𝜎2
gt + 𝐶2)
(17)
where 𝜇and 𝜎represent local means and standard de-
viations computed over 11 × 11 windows, with constants
𝐶1 = 0.012 and 𝐶2 = 0.032 preventing numerical instability
during division. This component primarily supervises in-
focus regions through direct pixel-wise comparison (L1)
and structural similarity preservation (SSIM).
Supervision for Out-of-Focus Regions. To specifically
supervise the physically-based defocus effects, we introduce
a dedicated loss term, dof. This loss is computed between
the ground-truth image convolved with our physics-based
kernel and the rendered image similarly convolved. It shares
the same structure as the in-focus loss but operates on the
defocused images:
(dof)
L1
=
1
|Ω|
∑
𝑝∈Ω
‖𝐼(dof)
rend (𝑝) −𝐼(dof)
gt
(𝑝)‖1
(18)
(dof)
SSIM = 1 −SSIM(𝐼(dof)
rend , 𝐼(dof)
gt
)
(19)
Composite Defocus Loss. These components are com-
bined into the final depth-of-field loss dof:
dof = (1 −𝜆(dof)
dssim)(dof)
L1
+ 𝜆(dof)
dssim(dof)
SSIM
(20)
This hierarchical structure allows for targeted optimiza-
tion of focus accuracy and defocus physics, which is crucial
for high-quality synthesis.
3.1.6. Depth-Aware Gaussian Density Control
Traditional 3D Gaussian reconstruction struggles in de-
focused regions due to uniform density allocation, which
limits the effectiveness of Gaussian points for downstream
tasks (Wu, Chen, Huang, Song and Zhang, 2024). Our
gradient-aware strategy adaptively modulates point density
using optical gradients derived from the CoC physics. The
proposed approach achieves three fundamental improve-
ments over conventional methods: (1) Edge preservation
through gradient-sensitive prioritization of points in depth
discontinuity regions, (2) Adaptive efficiency via automatic
density balancing based on quantile-based gradient statis-
tics, and (3) Physical consistency inherited from the optical
imaging model through CoC constraints. The control param-
eter 𝜏∈[0, 1] governs the quality-efficiency trade-off, where
higher 𝜏values enhance reconstruction fidelity at the cost
of increased computational resources, as formalized in the
preservation criterion.
Gradient Computation The CoC gradient magnitude
∇DoF is computed through differentiable rendering:
∇DoF = 𝜕dof
𝜕𝐱
(21)
the definition of dof refers to Section 3.1.5.
Adaptive Density Modulation A quantile-based preser-
vation criterion prioritizes structurally critical regions:
keep = 𝕀[‖∇DoF‖ ≥𝑄𝜏
({‖∇DoF‖})]
(22)
where 𝜏= 0.2 preserves the top 20% of points with the
highest gradients, and 𝑄𝜏denotes the 𝜏-th quantile.
Pruning Criterion The pruning logic combines opacity
thresholding with gradient-aware selection:
prune = (𝛼< 𝛼min)∨[(‖∇𝐱‖ < 𝑔min) ∧¬keep
] (23)
where 𝛼min and 𝑔min are empirically determined thresh-
olds for opacity and spatial gradients, respectively. The
logical operators ∨(OR), ∧(AND), and ¬ (NOT) implement
the compound pruning condition.
The base values for 𝛼min and 𝑔min are identical to those in
the original 3DGS framework ((Kerbl et al., 2023)), ensuring
a fair comparison. Our key contribution is the adaptive
criterion keep, which introduces a principled, physics-
aware mechanism that modulates the pruning process based
on geometric significance. It is important to note that our
method acts as a protective supplement to the original den-
sification logic, rather than adaptively lowering the global
threshold. As detailed in Appendix A.1, our experiments
confirm that simply tuning the global threshold is a fragile
strategy that can lead to training instability, whereas our
approach provides a stable performance improvement.
Gradient Statistics Interquartile-range weighting pre-
vents outlier dominance in gradient accumulation:
𝑤𝑖= exp
(
−
‖∇DoF,𝑖‖ −𝑄0.25
𝑄0.75 −𝑄0.25 + 𝜖
)
(24)
̂𝐺𝑖= ̂𝐺𝑖+ 𝑤𝑖‖∇DoF,𝑖‖
(25)
with 𝑄0.25 and 𝑄0.75 denoting the 25th and 75th per-
centiles of gradient magnitudes.
: Preprint submitted to Elsevier
Page 6 of 19

<!-- page 7 -->
3.2. Global Scale Recovery for Monocular Depth
Maps
3.2.1. Motivation for Scale Recovery
Monocular depth estimation methods (Almalioglu, Tu-
ran, Saputra, De Gusmão, Markham and Trigoni, 2022;
Liu, Zhang and Liu, 2024a) estimate depth values up to an
unknown scale factor for each image, resulting in metric
inconsistencies across multi-view observations. These scale
ambiguities lead to misaligned geometries when integrating
depth maps into 3DGS. Our scale recovery algorithm jointly
optimizes per-image parameters to ensure view-consistent
depth relationships, which are critical for realistic depth-of-
field synthesis.
3.2.2. Scale Recovery via Geometric Consistency
Given 𝑁images with camera parameters {(𝐊𝑖, 𝐑𝑖, 𝐭𝑖)}
and their corresponding raw monocular depth maps {𝐷𝑚},
we model the scaled depth at a pixel coordinate 𝐩as:
̃𝐷𝑚(𝐩) = 𝑠𝑖⋅𝐷𝑚(𝐩) + 𝑏𝑖
(26)
where 𝑠𝑖
>
0 and 𝑏𝑖are learnable scale and shift
parameters, respectively.
Feature Matching For image pairs (𝐼𝑖, 𝐼𝑗), we employ
the LoFTR descriptor to establish semi-dense local feature
correspondences.
Theoretical Depth Ratio For valid matches (𝐩𝑘
𝑖↔𝐩𝑘
𝑗),
we compute:
𝛾𝑘
𝑖𝑗= 𝐞⊤
𝑧
(
𝐑𝑗𝑖𝐱𝑘
𝑖+
𝐭𝑗𝑖
𝑍𝑘
𝑖
)
(27)
where 𝐱𝑘
𝑖= 𝐊−1
𝑖𝐩𝑘
𝑖and 𝑍𝑘
𝑖≈̃𝐷𝑚(𝐩𝑘
𝑖).
Joint Optimization We define an objective function that
is minimized over the set of all scale and shift parameters
{𝑠𝑖, 𝑏𝑖}𝑁
𝑖=1 across all images. The function combines a re-
projection term and a depth ratio consistency term:
𝑙𝑜𝑠𝑠=
∑
(𝑖,𝑗,𝑘)
‖‖‖𝜋(𝐑𝑗̃𝐗𝑘
𝑖+ 𝐭𝑗) −𝐩𝑘
𝑗
‖‖‖
2
⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟
Reprojection
+ 𝜆
∑
(𝑖,𝑗,𝑘)
(𝐷𝑚(𝐩𝑘
𝑗)
𝐷𝑚(𝐩𝑘
𝑖)
−𝛾𝑘
𝑖𝑗
)2
⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟
Ratio Consistency
(28)
where ̃𝐗𝑘
𝑖= ̃𝐷𝑘
𝑚𝐱𝑘
𝑖and 𝜆= 0.5. This optimization yields
the optimal parameters {𝑠∗
𝑖, 𝑏∗
𝑖}, which represent the globally
consistent scale and shift for each view.
Depth Alignment We apply the optimized parameters
{𝑠∗
𝑖, 𝑏∗
𝑖} to the raw monocular depths to obtain the final,
globally aligned depth maps {𝐷𝑎}:
𝐷𝑎(𝐩) = 𝑠∗
𝑖𝐷𝑚(𝐩) + 𝑏∗
𝑖
(29)
The aligned depths {𝐷𝑎} from Equation (29) enable
consistent blur synthesis through 3DGS as described in
Section 3.1.4.
3.3. Geometric Consistency Supervision via
Feature Matching
3.3.1. Depth Rendering with 3DGS
For a given viewpoint, the rendered depth value 𝐷𝑟(𝐱) at
pixel coordinate 𝐱is computed through alpha compositing
of 𝐾ordered Gaussians:
𝐷𝑟(𝐱) =
𝐾
∑
𝑘=1
𝛼𝑘𝑇𝑘𝑑𝑘
(30)
where 𝑇𝑘
=
∏𝑘−1
𝑙=1 (1 −𝛼𝑙) denotes the cumulative
transmittance up to the 𝑘-th Gaussian, 𝛼𝑘is the opacity of
the 𝑘-th Gaussian, and 𝑑𝑘represents the depth value of the
Gaussian center.
Note that this formulation is the standard for depth
rendering via alpha compositing, analogous to the color
rendering equation in the original 3DGS framework (Kerbl
et al., 2023). The sum of weights, ∑𝛼𝑘𝑇𝑘, represents the
total accumulated alpha. For the opaque surfaces targeted by
our reconstruction, the optimization process naturally drives
this sum to converge to a value near 1. Deviations from 1
are physically meaningful, representing rays viewing empty
space or semi-transparent geometry.
3.3.2. Cross-view Feature Matching
For image pairs (𝐼𝑖, 𝐼𝑗), we extract semi-dense corre-
spondences using a pre-trained LoFTR model:
𝑖𝑗= {(𝐱(𝑚)
𝑖
, 𝐱(𝑚)
𝑗
)}𝑀
𝑚=1
(31)
where matches are filtered by a confidence threshold
𝜏= 0.5.
3.3.3. Geometric Consistency Loss
As shown in Figure 5, for each matched pair (𝐱(𝑚)
𝑖
, 𝐱(𝑚)
𝑗
) ∈
𝑖𝑗:
1) Project 2D matches to 3D space:
𝐏(𝑚)
𝑖
= 𝜋−1
𝑖(𝐱(𝑚)
𝑖
, 𝐷𝑟(𝐱(𝑚)
𝑖
))
(32)
𝐏(𝑚)
𝑗
= 𝜋−1
𝑗(𝐱(𝑚)
𝑗
, 𝐷𝑟(𝐱(𝑚)
𝑗
))
(33)
2) Compute position discrepancy:
geo =
1
||
𝑀
∑
𝑚=1
‖𝐏(𝑚)
𝑖
−𝐏(𝑚)
𝑗‖1
(34)
3.4. Local Scale Restoration
Accurate depth estimation constitutes a fundamental re-
quirement for effectively harnessing depth-of-field effects
to strengthen structural supervision within 3D Gaussian
Splatting. While monocular depth estimation offers initial
depth cues, its inherent scale ambiguity prevents direct ap-
plication. To address this limitation, we propose a local scale
restoration framework through adaptive regional analysis.
: Preprint submitted to Elsevier
Page 7 of 19

<!-- page 8 -->
Figure 5: Schematic Diagram of Geometric Consistency Loss
Calculation: Feature points matched in two views are projected
into 3D space using rendered depth, and geometric constraints
are imposed by minimizing the distance between these pro-
jected points.
3.4.1. Grid-based Local Regions
Given a rendered depth map 𝐷𝑟∈ℝ𝐻×𝑊and monoc-
ular depth map 𝐷𝑚∈ℝ𝐻×𝑊, we decompose the image
domain into a grid of local regions. To ensure our method is
robust to varying image resolutions, we employ an adaptive
strategy. Instead of fixing the number of grid cells (ℎ, 𝑤),
we constrain the size of each cell, (𝑔ℎ, 𝑔𝑤), to be within an
empirically established range [𝑔min, 𝑔max] (15 to 60 pixels).
This range balances two factors: cells must be large enough
for stable parameter estimation (𝑔min) but small enough to
capture local depth variations (𝑔max). The number of cells
(ℎ, 𝑤) is then derived from the image dimensions (𝐻, 𝑊) to
meet these size constraints. Integer division is used to tile the
grid, ensuring all boundary pixels are included in the final
row and column of cells without clipping.
To guarantee robust parameter estimation, each grid cell
must maintain a minimum of 5 valid feature points. This
density constraint prevents numerical instability during local
linear transformation computation, particularly addressing
challenges from sparse or non-uniform feature distributions.
3.4.2. Local Linear Transformation
Within each grid cell 𝐺𝑖,𝑗, we establish a parametric
mapping between rendered depth 𝐷𝑟and monocular depth
𝐷𝑚through:
𝐷(𝑖,𝑗)
𝑟
= 𝑠𝑖,𝑗𝐷(𝑖,𝑗)
𝑚
+ 𝑡𝑖,𝑗,
(35)
where 𝑠𝑖,𝑗(scale factor) and 𝑡𝑖,𝑗(translation offset) denote
grid-specific transformation parameters. These parameters
are optimized via Tikhonov-regularized least squares min-
imization:
min
𝑠𝑖,𝑗,𝑡𝑖,𝑗
∑
𝑝∈𝐺𝑖,𝑗
‖‖‖𝐷𝑟(𝑝) −(𝑠𝑖,𝑗𝐷𝑚(𝑝) + 𝑡𝑖,𝑗
)‖‖‖
2
2 +𝜆
‖‖‖‖‖
[𝑠𝑖,𝑗
𝑡𝑖,𝑗
]‖‖‖‖‖
2
2
,
(36)
where 𝑝indexes valid feature points within the grid cell,
and 𝜆= 10−6 serves as the regularization coefficient to
ensure numerical stability in ill-posed conditions.
The closed-form solution derives from the normal equa-
tions formulation:
[𝑠𝑖,𝑗
𝑡𝑖,𝑗
]
= (𝑋⊤𝑋+ 𝜆𝐼)−1 𝑋⊤𝑦,
(37)
with design matrix 𝑋= [𝐷𝑚(𝑝)
𝟏] containing monoc-
ular depth measurements and an intercept term, and obser-
vation vector 𝑦= 𝐷𝑟(𝑝) comprising rendered depth values.
3.4.3. Depth Error Map Generation and Visualization
The depth error 𝐸𝑖,𝑗for each grid cell is computed as the
mean absolute difference between the rendered depth values
and their corresponding transformed monocular depth esti-
mates:
𝐸𝑖,𝑗=
1
|𝐺𝑖,𝑗|
∑
𝑝∈𝐺𝑖,𝑗
|||𝐷𝑟(𝑝) −(𝑠𝑖,𝑗𝐷𝑚(𝑝) + 𝑡𝑖,𝑗
)||| , (38)
where |𝐺𝑖,𝑗| represents the number of valid points in grid
cell 𝐺𝑖,𝑗, 𝐷𝑟(𝑝) denotes the rendered depth at point 𝑝, and
𝑠𝑖,𝑗𝐷𝑚(𝑝) + 𝑡𝑖,𝑗is the transformed monocular depth using
the optimized local linear transformation parameters.
These grid-level errors are then interpolated to the orig-
inal image resolution to form a comprehensive depth error
map 𝐸∈ℝ𝐻×𝑊. To enhance visualization contrast and
handle grid boundaries and regions with insufficient feature
points, we apply the following min-max normalization:
̂𝐸(𝑝) =
{ 𝐸(𝑝)−𝐸min
𝐸max−𝐸min ,
if 𝐸(𝑝) ≠1 and 𝐸(𝑝) is finite,
1,
otherwise.
(39)
where 𝐸min and 𝐸max represent the minimum and max-
imum error values among all valid pixels, respectively. The
default value of 1 is preserved for pixels in invalid regions
or grid cells with insufficient feature points.
To quantitatively analyze the spatial distribution char-
acteristics of depth estimation errors, Figure 6 provides a
multi-modal comparison comprising three aligned represen-
tations: (a) Ground truth RGB image, (b) Monocular depth
estimation results, and (c) Corresponding disparity error
heatmap generated by our method.
This multi-view visualization quantitatively reveals spa-
tial error concentration patterns and pinpoints challenging
regions characterized by complex geometric configurations,
occlusion boundaries, and fine-scale structural details.
3.4.4. Depth Consistency Loss
We design a depth consistency loss function to optimize
the alignment between rendered and monocular depths:
depth = abs + 𝛼corr,
(40)
: Preprint submitted to Elsevier
Page 8 of 19

<!-- page 9 -->
(a) Ground Truth
(b) Monocular Estimation
(c) Error Distribution
Figure 6: Comparative visualization of depth estimation components: (a) Ground truth, (b) Monocular depth prediction from our
proposed method, (c) Spatial error mapping. Error is computed only in regions with sufficient feature matches (non-yellow areas)
and normalized to [0, 1), where cooler colors indicate lower error. The ’N/A’ label on the color bar denotes regions where error
was not computed.
where 𝛼is a balancing coefficient empirically determined to
regulate the influence of correlation error.
where the absolute error term abs penalizes deviations
in valid regions:
abs =
1
|Ωvalid|
∑
𝑝∈Ωvalid
̂𝐸(𝑝),
(41)
with valid regions Ωvalid defined as pixels where both
error map values are non-zero and both rendered and monoc-
ular depths are positive.
The correlation error term corr enforces consistency
in invalid regions using min-max normalized depth maps:
corr =
||||||
1 −
1
|Ωinvalid|
∑
𝑝∈Ωinvalid
̂𝐷𝑟(𝑝) ̂𝐷𝑚(𝑝)
||||||
,
(42)
where Ωinvalid = {𝑝∈Ω|𝑝∉Ωvalid} is the complement
of the valid region, and ̂𝐷𝑟and ̂𝐷𝑚are min-max normalized
to [0, 1] as follows:
̂𝐷(𝑝) =
𝐷(𝑝) −𝐷min
𝐷max −𝐷min + 𝜖,
(43)
with 𝜖= 10−8 added for numerical stability.
The proposed depth consistency loss addresses three
critical challenges in depth alignment. First, the explicit min-
max normalization of rendered and monocular depth maps
( ̂𝐷𝑟and ̂𝐷𝑚) to the unit interval inherently handles scale
variations between different regions, eliminating the need
for ad-hoc calibration. Second, the dual-term design (abs
for valid regions and corr for invalid regions) ensures ro-
bustness against uneven feature point distributions by decou-
pling the optimization constraints. Finally, the integration of
normalized correlation in corr and the previously computed
error map ̂𝐸(𝑝) in abs jointly improve numerical stability,
effectively handling regions with insufficient valid matches
while maintaining consistent depth relationships. These at-
tributes are achieved with only a single hyperparameter 𝛼to
balance the two loss terms.
3.5. Total Loss Formulation
The overall optimization objective is formulated as a
weighted combination of four key components:
total = rgb + dof + 𝜆geogeo + 𝜆depthdepth
(44)
where rgb ensures photometric accuracy through RGB
reconstruction error minimization, dof enforces optical
consistency by aligning depth-of-field effects, geo main-
tains multi-view geometric consistency across adjacent frames
through feature matching, and depth aligns rendered depths
with monocular depth priors using our proposed depth
consistency loss.
The balancing weights 𝜆geo and 𝜆depth control the relative
influence of geometric and depth constraints, respectively.
This formulation enables joint optimization of appearance,
geometry, and optical properties within a unified framework,
effectively leveraging both learned monocular depth priors
and multi-view geometric constraints.
4. Experiment
4.1. Datasets
Our experiments were conducted on four datasets: Waymo
(Sun, Kretzschmar, Dotiwalla, Chouard, Patnaik, Tsui, Guo,
Zhou, Chai, Caine, Vasudevan, Han, Ngiam, Zhao, Timo-
feev, Ettinger, Krivokon, Gao, Joshi, Zhang, Shlens, Chen
and Anguelov, 2020), Mip-NeRF360 (Barron et al., 2022),
SS3DM (Hu, Wen, Zhou, Guo and Liu, 2024b), and the
YouTube (Cheng, Long, Yang, Yao, Yin, Ma, Wang and
Chen, 2024b) dataset. Waymo dataset is a large-scale urban
dataset. Mip-NeRF360 is a common NeRF benchmark. The
detailed descriptions and experimental results for both the
SS3DM and YouTube datasets will be provided in the 4.3.3.
4.2. Implementation Details
Our implementation extends the original 3DGS frame-
work (Kerbl et al., 2023) by incorporating enhanced depth-
of-field controls while maintaining compatibility for fair
comparisons. All experiments were conducted using the
: Preprint submitted to Elsevier
Page 9 of 19

<!-- page 10 -->
Table 1
Revised comparison with state-of-the-art methods on Waymo and Mip-NeRF 360 datasets. Our method shows significant gains
over the baseline through principled adaptation. Ours (Default) refers to our default configuration from the original manuscript.
Ours (All) refers to the full framework with tuned supervision weights. Ours (Best) reports the per-scene best PSNR achieved
across all experimental configurations, demonstrating the peak performance potential of our framework under this metric.
Method
Waymo
Mip-NeRF 360
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Instant-NGP (Müller et al., 2022)
30.98
0.886
0.281
25.59
0.699
0.331
Mip-NeRF 360 (Barron et al., 2022)
30.09
0.909
0.262
27.69
0.792
0.237
Zip-NeRF (Barron et al., 2023)
34.22
0.939
0.205
28.54
0.828
0.189
3DGS (Kerbl et al., 2023)
34.04
0.942
0.224
27.21
0.815
0.214
GaussianPro (Cheng et al., 2024a)
34.37
0.945
0.210
27.92
0.825
0.208
ConsistentGaussian (Ours, Default)
35.17
0.950
0.205
27.71
0.824
0.197
ConsistentGaussian (Ours, All)
-
-
-
27.92
0.827
0.189
ConsistentGaussian (Ours, Best)
-
-
-
27.95
0.826
0.195
original 3DGS configuration over 30,000 iterations on NVIDIA
GeForce RTX 4090 GPUs.
Key optical parameters adhere to standard photographic
configurations: a 50mm focal length (equivalent to a stan-
dard lens), an 𝑓∕5.6 aperture (providing a moderate depth
of field), and a 36mm full-frame sensor. To balance defocus
realism with rendering performance, we limited the maxi-
mum blur kernel size to 7 × 7 pixels.
When the Dynamic Focus Strategy is activated, the
focus distance 𝑑𝑓is optimized through statistical analysis
of depth distributions. Empirical evaluation demonstrates
that both median depth (𝑑med) and first-tercile depth (𝑑1∕3)
exhibit comparable performance metrics. The median depth
configuration is adopted as the default selection to maximize
computational efficiency.
Our depth-aware density control preserves the top 20%
of Gaussians by applying a 𝜏= 0.2 quantile threshold to
depth gradients, thereby prioritizing structural regions while
maintaining computational efficiency.
4.3. Results
To rigorously validate our approach, we established
comparative benchmarks against five state-of-the-art novel
view synthesis methods: GaussianPro (Cheng et al., 2024a),
Instant-NGP (Müller et al., 2022), Mip-NeRF 360 (Barron
et al., 2022), Zip-NeRF (Barron et al., 2023), and the base-
line 3DGS (Kerbl et al., 2023). The quantitative performance
comparison between our method and these approaches is
systematically presented in Table 1.
4.3.1. Quantitative Comparisons
Table 1 presents the comprehensive evaluation of our
method against state-of-the-art approaches on Waymo and
Mip-NeRF 360 datasets, analyzed through three key metrics:
peak signal-to-noise ratio (PSNR), structural similarity in-
dex measure (SSIM) (Wang, Bovik, Sheikh and Simoncelli,
2004), and the learned perceptual image patch similarity
(LPIPS) (Zhang, Isola, Efros, Shechtman and Wang, 2018).
The experimental results for GaussianPro are sourced di-
rectly from its original implementation.
Analysis on Unbounded Scenes and Adaptive Super-
vision. The Mip-NeRF 360 dataset, comprising nine distinct
scenes (five outdoor and four indoor) that present diverse
and challenging unbounded environments, serves as an ideal
testbed for analyzing the adaptability of supervision strate-
gies. Unlike the more structurally homogeneous forward-
facing scenes in Waymo, its diversity (e.g., from the struc-
tured ’room’ to the texture-rich ’treehill’) motivates a deeper,
scene-specific investigation into the interplay between our
framework’s components and scene characteristics. Our ini-
tial results with default parameters, while robust on Waymo,
were less conclusive on this dataset, prompting this focused
analysis.
As shown in the revised Table 1, our full framework with
tuned weights (Ours, All) improves the average PSNR to
27.92 dB. Furthermore, by selecting the configuration that
yielded the best PSNR for each scene (Ours, Best), the
average PSNR is further lifted to a peak of 27.95 dB. While
this does not surpass the specialized Zip-NeRF model, it
highlights a key insight: for scenes with unreliable geometric
priors due to repetitive textures (e.g., ‘treehill‘), a targeted
application of our supervision signals is most effective.
Conversely, for well-structured scenes (e.g., ‘room‘), the full
suite of losses yields substantial improvements. This demon-
strates that our framework’s value lies not only in its strong
baseline performance but also in its flexibility, providing a
toolkit and principled guidelines for practitioners to achieve
optimal results by adapting the supervision strategy to the
scene at hand.
Cross-Dataset Analysis The 7.25 dB PSNR disparity
between Waymo (35.17 dB) and Mip-NeRF 360 (27.92 dB)
reflects fundamental challenges in outdoor driving scenarios
with extensive depth ranges compared to indoor scenes with
constrained depth variation. Our methodology specifically
targets structural inconsistencies in distant views, which
contrasts with the Mip-NeRF 360 dataset’s characteristics
: Preprint submitted to Elsevier
Page 10 of 19

<!-- page 11 -->
Figure 7: Visual comparison on Waymo dataset. Our method achieves superior detail preservation in both mid-range and distant
scenes compared to 3DGS and GaussianPro.
where most sub-scenes exhibit limited depth variation. Fur-
thermore, our analysis identifies implementation discrepan-
cies between the Mip-NeRF 360 variants used in Gaussian-
Pro and our baseline experiments.
4.3.2. Qualitative Comparisons
Our method demonstrates significant visual improve-
ments in complex outdoor scenarios, as evidenced by com-
parative reconstructions on the Waymo dataset (Figure 7).
The key enhancement manifests in resolving the inherent
near-far supervision conflict through depth-aware physical
modeling, particularly evident in scenes containing both
: Preprint submitted to Elsevier
Page 11 of 19

<!-- page 12 -->
Table 2
Comparison on SS3DM, YouTube, and LibraryDoF datasets.
Method
SS3DM
YouTube
LibraryDoF
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
3DGS
30.47
0.891
0.253
34.98
0.960
0.081
23.13
0.729
0.340
GaussianPro
31.44
0.906
0.225
35.58
0.9645
0.071
22.62
0.731
0.330
Ours
33.83
0.928
0.192
36.50
0.972
0.057
24.81
0.810
0.241
3DGS
GaussianPro
Ours
Ground Truth
Figure 8: Comparison of model renderings on the LibraryDoF dataset among our method, GaussianPro , and 3DGS. Our method
shows distinct advantages, with more accurate detail restoration, and a more immersive visual experience.
close-range structural elements and distant environmental
features.
Geometric Consistency Enhancement As shown in
Figure 7 (red boxes), our approach maintains coherent ge-
ometric patterns across various depth layers where baseline
methods exhibit structural fragmentation or spatial artifacts.
This improvement stems from two synergistic mechanisms:
1) The depth-sensitive density control prioritizes representa-
tion capacity allocation to geometrically critical regions, and
2) The hybrid depth estimation framework ensures metric
consistency across multi-view observations through Equa-
tion 29. These technical components jointly address the dual
challenges of preserving near-field structural details while
maintaining far-field depth coherence.
Technical Implementation All visualizations strictly
adhere to the rendering configurations defined in Section 4.2,
with kernel parameterization following the specifications in
Section 3.1.2. Depth-dependent blur synthesis is governed
by the physical imaging model in Equation 1.
4.3.3. Additional experiments
We performed additional experiments on diverse datasets
to validate the generalizability of the ConsistentGaussian
model (ours). Beyond the widely adopted real-world Mip-
Nerf 360 and Waymo datasets in the 3D reconstruction field,
we specifically introduced three distinct data sources: the
SS3DM synthetic autonomous driving dataset (Hu et al.,
2024b), YouTube video sequences (Cheng et al., 2024a),
and our self-collected LibraryDoF dataset captured using
consumer-grade cameras. Through experimental validation
of these three types of differentiated datasets, we further
corroborated the robustness advantages of our proposed
method in handling multi-source heterogeneous data.
The SS3DM benchmark provides controlled-environment
evaluation for street view surface reconstruction, offering
CARLA-simulator-generated 3D ground-truth meshes that
enable precise geometric and photometric assessments. The
YouTube dataset, curated by GaussianPro, contains four dis-
tinct Subscenes extracted from publicly available YouTube
videos.
The SS3DM dataset contains eight virtual towns (Town01
- Town07, Town10), each featuring multiple sub-scenes. We
standardize our evaluation by selecting the 150_streetsurf
sub-scene in Town01 - Town07, while employing 200_streetsurf
in Town10 which lacks the 150_streetsurf configuration. This
strategy maintains the consistency of the evaluation while
accommodating the inherent variations in the dataset.
For the YouTube dataset, each sub-scene corresponds
to 360-degree aerial footage of iconic landmarks, such as
the Eiffel Tower. We utilized the YouTube dataset to further
validate the effectiveness of our proposed method.
To further evaluate the efficacy of our proposed method,
we meticulously acquired a comprehensive dataset using a
Canon R10 camera in front of the university library. The
dataset comprises two distinct categories of images with
approximately equal distribution and similar amounts: near-
all-in-focus images and shallow depth-of-field images. The
near-all-in-focus images were captured with the focus set
at distant objects, simulating an all-in-focus effect, while
the shallow depth-of-field images were obtained by shifting
the focal plane forward under identical camera extrinsic
parameters, thereby introducing defocus effects.
For model training implementation, we adhered to the
splitting strategy used in the original 3DGS, with a crucial
modification: we ensured that the training set contained
a balanced mixture of both near-all-in-focus and shallow
: Preprint submitted to Elsevier
Page 12 of 19

<!-- page 13 -->
Table 3
Per-scene performance comparison across SS3DM towns.
Method
Town01
Town02
Town03
Town04
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
3DGS
32.466
0.934
0.220
25.549
0.844
0.342
26.356
0.840
0.325
31.976
0.910
0.205
GaussianPro
32.425
0.929
0.220
31.588
0.918
0.233
31.101
0.880
0.250
31.713
0.913
0.202
Ours
34.400
0.953
0.178
34.000
0.947
0.184
35.903
0.944
0.168
34.199
0.934
0.172
Method
Town05
Town06
Town07
Town10
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
3DGS
33.350
0.927
0.196
32.305
0.906
0.254
30.220
0.870
0.244
31.562
0.895
0.241
GaussianPro
31.959
0.918
0.204
32.210
0.908
0.246
29.372
0.886
0.213
31.145
0.895
0.232
Ours
34.432
0.940
0.174
34.082
0.918
0.231
30.944
0.877
0.225
32.710
0.913
0.202
Table 4
Ablation study with component integration and blur kernel comparisons on Waymo dataset
Configuration
PSNR↑
SSIM↑
LPIPS↓
ΔPSNR
ΔSSIM
ΔLPIPS
3DGS (Baseline)
34.04
0.942
0.224
-
-
-
Progressive Component Integration
+ Depth-of-field (Gaussian Blur Kernel)
34.61
0.946
0.217
+0.57
+0.004
-0.007
+ Gradient Control
35.00
0.948
0.209
+0.96
+0.006
-0.015
+ Point Match Loss
34.98
0.949
0.206
+0.94
+0.007
-0.018
Full (Gaussian Blur Kernel)
35.17
0.950
0.205
+1.13
+0.008
-0.019
Blur Kernel Analysis
Full (Polygonal Blur Kernel)
35.03
0.948
0.209
+0.99
+0.006
-0.015
Full (SmoothStep Blur Kernel)
35.03
0.948
0.209
+0.99
+0.006
-0.015
No Depth-of-field Loss
34.42
0.944
0.221
+0.38
+0.002
-0.003
depth-of-field images, whereas the test set exclusively con-
sisted of near-all-in-focus images. During the loss com-
putation phase, the base reconstruction loss rgb is only
calculated when the depth-of-field strategy is disabled or the
loaded image is near all-in-focus. In contrast, for shallow
depth-of-field images, we computed dof. When training
comparative baseline models, we restricted the input to near-
all-in-focus images exclusively, as these approaches lack the
capability to process images with pronounced depth-of-field
effects.
The scene-specific evaluation results of the synthetic
SS3DM dataset are presented in Table 3, which provides
a comprehensive overview of performance metrics across
its distinct scenarios. Table 2 further summarizes the mean
metric comparisons among SS3DM (synthetic autonomous
driving data), YouTube sequences (real-world dynamic scenes),
and our LibraryDoF dataset (challenging handheld cap-
tures). Figure 8 shows some rendering results from the Li-
braryDoF dataset. These cross-dataset comparisons demon-
strate the consistent superiority of our method across diverse
data modalities, highlighting its robust generalization ca-
pability from synthetic to real-world scenarios and across
varying imaging conditions.
4.4. Ablation Study
4.4.1. Experimental Protocol
Our ablation analysis adopts two complementary strate-
gies: 1) Progressive integration of core components to isolate
individual contributions, and 2) Comparative evaluation of
blur kernel implementations. The first protocol sequentially
adds depth-of-field supervision, gradient-aware density con-
trol, point matching loss, and monocular depth alignment
to the 3DGS baseline. The second protocol substitutes the
Gaussian blur kernel with Polygonal blur kernel or Smooth-
Step blur kernel in the full configuration. The "No Depth-
of-field Loss" condition removes defocus supervision while
retaining other components, quantifying its geometric regu-
larization effect.
4.4.2. Component Effectiveness
The Gaussian blur-based depth-of-field supervision es-
tablishes the foundational improvement (+0.57 dB PSNR),
validating our physics-driven defocus modeling. Subsequent
integration of gradient-aware density control contributes
an additional +0.39 dB enhancement, demonstrating its
efficacy in preserving geometrically critical regions. The
marginal LPIPS improvement (0.206 vs 0.209) when adding
point matching loss indicates enhanced structural consis-
tency through multi-view correspondence constraints.
The full configuration achieves peak performance (35.17
dB PSNR), confirming the necessity of unified depth su-
pervision. The 0.75 dB degradation in the "No Depth-of-
field Loss" condition (34.42 dB vs 35.17 dB) quantitatively
demonstrates defocus supervision’s critical role in geometric
regularization, consistent with our theoretical analysis in
Section 3.1.1.
4.4.3. Blur Kernel Analysis
All blur kernel implementations significantly outper-
form the baseline, with the Gaussian configuration achiev-
ing optimal PSNR (35.17 dB) and LPIPS (0.205). The
Polygonal blur kernel generates physically accurate bokeh
effects through parametric aperture modeling (Figure 4(d)),
effectively replicating real camera optics. The Polygonal
: Preprint submitted to Elsevier
Page 13 of 19

<!-- page 14 -->
Table 5
Robustness analysis of the depth-gradient preservation quantile (𝜏) on the Mip-NeRF 360 dataset. PSNR↑is stable across a
wide range of 𝜏values, indicating low sensitivity. The maximum performance deviation for each scene is shown in parentheses,
confirming either "Highly Robust" (≤0.1 dB) or "Good Robustness" (0.1-0.2 dB) according to our protocol. Our default value is
𝜏= 0.2.
Scene
Preservation Quantile 𝜏
0.1
0.2
0.4
0.5
0.6
0.8
0.9
bicycle (Δ0.13 dB)
25.31
25.35
25.39
25.37
25.35
25.33
25.26
garden (Δ0.10 dB)
27.50
27.58
27.53
27.56
27.55
27.58
27.60
room (Δ0.14 dB)
31.99
31.96
32.09
32.03
32.08
31.98
31.95
and SmoothStep blur kernel implementations yield identical
performance.
Implementation Guidelines The Gaussian blur kernel
demonstrates superior overall performance across metrics,
making it the default choice for general applications. The
Polygonal blur kernel is recommended for scenarios requir-
ing optical realism, particularly when synthesizing aperture-
specific effects. The SmoothStep blur kernel provides en-
hanced edge preservation for high-frequency detail recov-
ery and is suitable for post-processing applications. This
parametric design framework maintains core reconstruction
performance while accommodating diverse optical require-
ments.
4.4.4. Hyperparameter Robustness
To address the practical usability of our method, we
analyze the sensitivity of the key new hyperparameter in-
troduced: the depth-gradient preservation quantile, 𝜏(see
Section 3.1.6). We evaluate the final reconstruction quality
(PSNR) on three diverse scenes from the Mip-NeRF 360
dataset while varying 𝜏over a wide range from 0.1 (preserv-
ing top 90% of gradients) to 0.9 (preserving top 10%). As
shown in Table 5, the performance is highly stable across all
scenes. The maximum PSNR deviation is less than 0.15 dB,
demonstrating that our method is not sensitive to the precise
choice of this parameter and that our default value (𝜏= 0.2)
is a robust choice for general use.
In summary, our framework is designed to be both prin-
cipled and practical, minimizing the need for extensive hy-
perparameter tuning and ensuring its reproducibility.
5. Conclusion
We propose a physics-guided framework that enhances
3D Gaussian Splatting through depth-of-field-induced geo-
metric supervision, addressing three fundamental challenges
in neural scene reconstruction. First, our differentiable de-
focus convolution model physically emulates camera op-
tics through parametric kernel design, achieving optically
faithful bokeh effects while preserving computational ef-
ficiency through separable convolution operators. Second,
our gradient-aware density control mechanism dynamically
preserves geometrically critical structures through quantile-
based pruning, particularly effective in maintaining urban
scene integrity. Third, hierarchical depth alignment inte-
grates global monocular depth estimation with local grid-
based corrections, significantly enhancing geometric con-
sistency. Comprehensive evaluations demonstrate state-of-
the-art performance in structured environments, showing
enhanced rendering fidelity and improved depth estimation
accuracy.
Current limitations in unbounded scene optimization
suggest three research directions: 1) Temporal focus adapta-
tion mechanisms for video-consistent dynamic scene recon-
struction, 2) depth-discontinuous kernel blending strategies
combining physically accurate polygonal blur kernel with
detail-preserving SmoothStep blur kernel, and 3) adaptive
optics simulations for extreme depth ranges using dynamic
kernel scaling.
A. Appendix
A.1. Analysis of Densification Strategy
To validate our adaptive preservation mechanism (see
Section 3.1.6), we compared it against the simpler alternative
of globally tuning the densification gradient threshold. As
shown in Table 6, this manual tuning is a fragile process.
Lowering the threshold from the 3DGS default of 0.0002
leads to training failure. In contrast, our adaptive method,
which operates on the same stable default threshold, not
only avoids this instability but also delivers a consistent
performance gain over the baseline. While it is possible to
achieve marginal gains by carefully tuning the threshold,
this requires a delicate, scene-specific process. Our method
provides a robust path to high performance without such
manual intervention.
A.2. Robustness of Empirical Parameters
Our framework introduces several parameters that are
empirically set. Here, we provide a detailed analysis to
demonstrate their robustness and validate our default choices.
A.2.1. Geometric Supervision Weights
The weights for the geometric consistency loss (𝜆geo)
and depth consistency loss (𝜆depth) are set to 0.05 and 0.005
by default. As shown in Table 7, these defaults are robust
for standard scenes like ‘bicycle‘, where performance varies
by only 0.04 dB across a 100x change in weights. For
pathological scenes with unreliable geometric priors like
: Preprint submitted to Elsevier
Page 14 of 19

<!-- page 15 -->
Table 6
Quantitative comparison of densification strategies on the Mip-NeRF 360 ‘bicycle‘ scene. Our method is compared against
baselines with our adaptive mechanism disabled.
Method
densify grad threshold
PSNR↑
Ours (Adaptive)
0.0002
25.35
Baseline (Default Threshold)
0.0002
25.26
Baseline (Lowered Threshold)
≤0.00015
Training Failed
Table 7
Robustness of geometric supervision weights (𝜆) on Mip-NeRF 360 scenes. PSNR↑is stable for standard scenes, while pathological
scenes benefit from principled attenuation.
Scene
Default Weights (1x)
Attenuated (10x)
Attenuated (100x)
bicycle
25.35
25.31
25.35
bonsai
32.44
32.64
32.80
Table 8
Comparison of our adaptive grid strategy vs. a range of fixed
grid sizes on Mip-NeRF 360 scenes. The optimal fixed grid
size (bolded) is highly scene-dependent. Our adaptive method
consistently achieves near-optimal performance automatically.
Performance is measured in PSNR↑.
Grid Strategy
bicycle
garden
room
Ours (Adaptive)
25.35
27.59
32.09
8x8
25.33
27.58
32.10
16x16
25.34
27.50
32.01
32x32
25.37
27.57
32.08
64x64
25.38
27.53
31.99
128x128
25.34
27.54
31.97
‘bonsai‘, our analysis confirms the effectiveness of a key
adaptive criterion: strategically reducing the weights leads
to significant performance gains. This provides a clear, prin-
cipled guideline for tuning in such specific cases, rather than
requiring blind experimentation.
A.2.2. Adaptive Grid Strategy Parameters
Our adaptive grid strategy constrains the cell size to an
empirically established range of [𝑔min, 𝑔max] = [15, 60] pix-
els. The rationale for this effective range is rooted in the bias-
variance tradeoff. We validate this choice with two analyses.
First, Table 8 shows that the optimal fixed grid size is highly
scene-dependent, making manual tuning impractical. Our
adaptive method consistently performs competitively against
the best fixed grid in each case, automating this choice.
Second, Table 9 shows the method’s low sensitivity to the
precise values of 𝑔min and 𝑔max, confirming our default range
is a robust choice.
A.2.3. Maximum Blur Kernel Size
Our analysis of the ’maximum blur kernel size’, a key
parameter in our defocus model, reveals a nuanced but
important finding, as shown in Table 10. We compared our
default size of 7x7 against a smaller 3x3 kernel across four
Table 9
Sensitivity analysis of the adaptive grid boundaries [𝑔min, 𝑔max]
on the ‘bicycle‘ scene. PSNR↑is highly stable around our
default setting of [15, 60].
[𝑔min, 𝑔max]
PSNR↑
[5, 50]
25.26
[5, 70]
25.32
[15, 50]
25.37
[15, 60]
25.35
[25, 50]
25.37
[25, 70]
25.30
Table 10
Sensitivity analysis of maximum blur kernel size on four
diverse Mip-NeRF 360 scenes. A smaller 3x3 kernel consistently
provides either superior or statistically equivalent performance.
Performance is measured in PSNR↑.
Scene
Kernel Size
PSNR↑
bicycle
3x3
25.38
7x7
25.35
flowers
3x3
22.01
7x7
22.02
garden
3x3
27.55
7x7
27.53
room
3x3
32.10
7x7
31.87
diverse scenes. The results indicate that a smaller kernel
is a consistently strong choice. For scenes with prominent
fine structures like ’room’ and ’bicycle’, the 3x3 kernel
provides a modest but measurable performance benefit, im-
proving PSNR by +0.23 dB and +0.03 dB, respectively. For
other scenes such as ’garden’ and ’flowers’, the performance
is highly robust to the kernel size, with negligible differ-
ences between the two settings. This comprehensive analysis
validates our default 7x7 as a robust baseline, while also
providing a clear, data-driven guideline for practitioners:
: Preprint submitted to Elsevier
Page 15 of 19

<!-- page 16 -->
3DGS
GaussianPro
Ours
Ground Truth
Figure 9: Qualitative comparison on diverse Mip-NeRF 360 scenes. (a) In the ’flowers’ scene, our method reconstructs both
intricate petal structures and distant foliage with significantly higher textural fidelity and fewer artifacts. (b) In the ’room’ scene,
our method excels at rendering fine geometric details, such as the cabinet handle, while producing sharper textures on the
bookshelf and floor. (c) In the ’garden’ scene, our method achieves a more complete reconstruction of complex objects (e.g.,
the bucket) and fine foliage, with a clear reduction in artifacts. (d) In the ’bicycle’ scene, our method dramatically improves the
realism of the ground plane, rendering grass and soil with superior detail and textural accuracy.
for maximizing fidelity, especially in scenes with intricate
geometry, selecting a smaller 3x3 kernel is a principled and
effective optimization.
A.3. Additional Qualitative Comparisons on
Mip-NeRF 360
To provide further visual evidence of our method’s per-
formance on diverse and challenging unbounded scenes,
: Preprint submitted to Elsevier
Page 16 of 19

<!-- page 17 -->
3DGS
GaussianPro
Ours
Ground Truth
Town01
Town02
Town03
Town04
Figure 10: Qualitative comparison on the synthetic SS3DM dataset. Our method consistently achieves superior visual quality
across various scenes. In Town01, the proposed method achieved higher fidelity in the reconstruction of building facades and
road textures. In Town02, our method rendered distant architecture with enhanced realism and detail, particularly at road corners
characterized by sparse views. In Town03, under complex lighting, our approach preserved sharp structural details on buildings
and roadside signs, mitigating the blurring artifacts observed in baseline results. In Town04, our method generated more coherent
geometry and clearer details for intricate elements, including road textures at sparsely-viewed corners and distant trees.
Figure 9 presents qualitative comparisons against state-of-
the-art methods on three scenes from the Mip-NeRF 360
dataset. These visualizations complement the quantitative
results in Table 1 and demonstrate our method’s superior
ability to reconstruct complex geometry and fine details.
A.4. Additional Qualitative Results on Diverse
Datasets
To further demonstrate the generalization capabilities
of our framework, this section provides qualitative com-
parisons on the synthetic SS3DM dataset. As illustrated in
Figure 10, our method consistently produces reconstructions
with higher fidelity and fewer artifacts compared to baseline
methods.
CRediT authorship contribution statement
Yu Deng: Methodology, Software, Validation, Formal
analysis, Investigation, Resources, Data Curation, Writing -
Original Draft, Writing - Review & Editing, Visualization.
Baozhu Zhao: Conceptualization, Methodology, Investiga-
tion, Resources, Writing - Original Draft, Writing - Review
& Editing. Junyan Su: Validation, Formal analysis, Writing
- Review & Editing. Xiaohan Zhang: Writing - Review &
: Preprint submitted to Elsevier
Page 17 of 19

<!-- page 18 -->
Editing. Qi Liu: Supervision, Writing - Review & Editing,
Funding acquisition.
References
Almalioglu, Y., Turan, M., Saputra, M.R.U., De Gusmão, P.P., Markham,
A., Trigoni, N., 2022. Selfvio: Self-supervised deep monocular visual–
inertial odometry and depth estimation. Neural Networks 150, 119–136.
Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.,
2022.
Mip-nerf 360: Unbounded anti-aliased neural radiance fields,
in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 5470–5479.
Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P., 2023.
Zip-nerf: Anti-aliased grid-based neural radiance fields, in: Proceedings
of the IEEE/CVF International Conference on Computer Vision, pp.
19697–19705.
Blanc, H., Deschaud, J.E., Paljic, A., 2024. Raygauss: Volumetric gaussian-
based ray casting for photorealistic novel view synthesis. URL: https:
//arxiv.org/abs/2408.03356, arXiv:2408.03356.
Campbell, N.D., Vogiatzis, G., Hernández, C., Cipolla, R., 2008. Using
multiple hypotheses to improve depth-maps for multi-view stereo, in:
Computer Vision–ECCV 2008: 10th European Conference on Computer
Vision, Marseille, France, October 12-18, 2008, Proceedings, Part I 10,
Springer. pp. 766–779.
Chen, G., Wang, W., 2024.
A survey on 3d gaussian splatting.
arXiv
preprint arXiv:2401.03890 .
Cheng, K., Long, X., Yang, K., Yao, Y., Yin, W., Ma, Y., Wang, W., Chen,
X., 2024a. Gaussianpro: 3d gaussian splatting with progressive propa-
gation, in: Forty-first International Conference on Machine Learning.
Cheng, K., Long, X., Yang, K., Yao, Y., Yin, W., Ma, Y., Wang, W., Chen,
X., 2024b. Gaussianpro: 3d gaussian splatting with progressive prop-
agation, in: Forty-first International Conference on Machine Learning,
ICML 2024, Vienna, Austria, July 21-27, 2024, OpenReview.net.
Cui, Y., Knoll, A., 2024. Dual-domain strip attention for image restoration.
Neural Networks 171, 429–439.
Engel, J., Schöps, T., Cremers, D., 2014.
Lsd-slam: Large-scale direct
monocular slam, in: European conference on computer vision, Springer.
pp. 834–849.
Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., 2023. Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps.
arXiv preprint arXiv:2311.17245 .
Feng, Z., Yang, L., Guo, P., Li, B., 2023.
Cvrecon: Rethinking 3d
geometric feature learning for neural reconstruction, in: Proceedings
of the IEEE/CVF International Conference on Computer Vision, pp.
17750–17760.
Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa,
A., 2022. Plenoxels: Radiance fields without neural networks, in: Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 5501–5510.
Furukawa, Y., Hernández, C., et al., 2015. Multi-view stereo: A tutorial.
Foundations and Trends® in Computer Graphics and Vision 9, 1–148.
Giang, K.T., Song, S., Jo, S., 2021.
Curvature-guided dynamic scale
networks for multi-view stereo. arXiv preprint arXiv:2112.05999 .
Girish, S., Gupta, K., Shrivastava, A., 2024. Eagles: Efficient accelerated
3d gaussians with lightweight encodings, in: European Conference on
Computer Vision, Springer. pp. 54–71.
Gu, X., Fan, Z., Zhu, S., Dai, Z., Tan, F., Tan, P., 2020.
Cascade
cost volume for high-resolution multi-view stereo and stereo matching,
in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 2495–2504.
Henzler, P., Mitra, N.J., Ritschel, T., 2019.
Escaping plato’s cave: 3d
shape from adversarial rendering, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 9984–9993.
Hu, M., Yin, W., Zhang, C., Cai, Z., Long, X., Chen, H., Wang, K., Yu,
G., Shen, C., Shen, S., 2024a.
Metric3d v2: A versatile monocular
geometric foundation model for zero-shot metric depth and surface
normal estimation. arXiv preprint arXiv:2404.15506 .
Hu, Y., Wen, K., Zhou, H., Guo, X., Liu, Y., 2024b. SS3DM: benchmarking
street-view surface reconstruction with a synthetic 3d mesh dataset,
in: Advances in Neural Information Processing Systems 38: Annual
Conference on Neural Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15, 2024.
Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S., 2024. 2d gaussian splatting
for geometrically accurate radiance fields, in: ACM SIGGRAPH 2024
conference papers, pp. 1–11.
Ji, M., Gall, J., Zheng, H., Liu, Y., Fang, L., 2017. Surfacenet: An end-to-
end 3d neural network for multiview stereopsis, in: Proceedings of the
IEEE international conference on computer vision, pp. 2307–2315.
Kaneko, T., 2022. Ar-nerf: Unsupervised learning of depth and defocus
effects from natural images with aperture rendering neural radiance
fields, in: Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 18387–18397.
Kar, A., Häne, C., Malik, J., 2017. Learning a multi-view stereo machine.
Advances in neural information processing systems 30.
Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3d gaussian
splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
139–1.
Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., Drettakis,
G., 2024.
A hierarchical 3d gaussian representation for real-time
rendering of very large datasets. ACM Transactions on Graphics (TOG)
43, 1–15.
Lee, D., Lee, M., Shin, C., Lee, S., 2023. Dp-nerf: Deblurred neural radi-
ance field with physical scene priors, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 12386–
12396.
Lee, J.C., Rho, D., Sun, X., Ko, J.H., Park, E., 2024. Compact 3d gaussian
representation for radiance field, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 21719–
21728.
Li, J., Shi, Y., Cao, J., Ni, B., Zhang, W., Zhang, K., Van Gool, L., 2024.
Mipmap-gs: Let gaussians deform with scale-specific mipmap for anti-
aliasing rendering. arXiv preprint arXiv:2408.06286 .
Liu, X., Zhang, T., Liu, M., 2024a.
Joint estimation of pose, depth,
and optical flow with a competition–cooperation transformer network.
Neural Networks 171, 263–275.
Liu, Y., Luo, C., Fan, L., Wang, N., Peng, J., Zhang, Z., 2024b. Citygaus-
sian: Real-time high-quality large-scale scene rendering with gaussians,
in: European Conference on Computer Vision, Springer. pp. 265–282.
Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.,
2022a.
Deblur-nerf: Neural radiance fields from blurry images, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 12861–12870.
Ma, Z., Teed, Z., Deng, J., 2022b. Multiview stereo with cascaded epipolar
raft, in: European Conference on Computer Vision, Springer. pp. 734–
750.
Mai, A., Hedman, P., Kopanas, G., Verbin, D., Futschik, D., Xu, Q., Kuester,
F., Barron, J.T., Zhang, Y., 2024.
Ever: Exact volumetric ellipsoid
rendering for real-time view synthesis. arXiv preprint arXiv:2410.01804
.
Mildenhall, B., Hedman, P., Martin-Brualla, R., Srinivasan, P.P., Barron,
J.T., 2022. Nerf in the dark: High dynamic range view synthesis from
noisy raw images, in: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 16190–16199.
Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi,
R., Ng, R., 2021. Nerf: Representing scenes as neural radiance fields for
view synthesis. Communications of the ACM 65, 99–106.
Müller, T., Evans, A., Schied, C., Keller, A., 2022. Instant neural graphics
primitives with a multiresolution hash encoding. ACM transactions on
graphics (TOG) 41, 1–15.
Mur-Artal, R., Montiel, J.M.M., Tardos, J.D., 2015. Orb-slam: a versatile
and accurate monocular slam system. IEEE transactions on robotics 31,
1147–1163.
Radl, L., Steiner, M., Parger, M., Weinrauch, A., Kerbl, B., Steinberger, M.,
2024. Stopthepop: Sorted gaussian splatting for view-consistent real-
time rendering. ACM Transactions on Graphics (TOG) 43, 1–17.
: Preprint submitted to Elsevier
Page 18 of 19

<!-- page 19 -->
Schonberger, J.L., Frahm, J.M., 2016. Structure-from-motion revisited, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 4104–4113.
Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., Zollhofer,
M., 2019.
Deepvoxels: Learning persistent 3d feature embeddings,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 2437–2446.
Snavely, N., Seitz, S.M., Szeliski, R., 2006. Photo tourism: exploring photo
collections in 3d, in: ACM siggraph 2006 papers, pp. 835–846.
Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui,
P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W.,
Ngiam, J., Zhao, H., Timofeev, A., Ettinger, S., Krivokon, M., Gao,
A., Joshi, A., Zhang, Y., Shlens, J., Chen, Z., Anguelov, D., 2020.
Scalability in perception for autonomous driving: Waymo open dataset,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
Wang, C., Wolski, K., Kerbl, B., Serrano, A., Bemana, M., Seidel, H.P.,
Myszkowski, K., Leimkühler, T., 2024a. Cinematic gaussians: Real-time
hdr radiance fields with depth of field, in: Computer Graphics Forum,
Wiley Online Library. p. e15214.
Wang, P., Liu, Y., Chen, Z., Liu, L., Liu, Z., Komura, T., Theobalt, C., Wang,
W., 2023. F2-nerf: Fast neural radiance field training with free camera
trajectories, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 4150–4159.
Wang, Y., Chakravarthula, P., Chen, B., 2024b. Dof-gs: Adjustable depth-
of-field 3d gaussian splatting for refocusing, defocus rendering and blur
removal. arXiv preprint arXiv:2405.17351 .
Wang, Y., He, X., Peng, S., Tan, D., Zhou, X., 2024c. Efficient loftr: Semi-
dense local feature matching with sparse-like speed, in: Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, pp. 21666–21675.
Wang, Y., Yang, S., Hu, Y., Zhang, J., 2022. Nerfocus: Neural radiance field
for 3d synthetic defocus. arXiv preprint arXiv:2203.05189 .
Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P., 2004.
Image
quality assessment: from error visibility to structural similarity. IEEE
transactions on image processing 13, 600–612.
Wu, Y., Chen, X., Huang, X., Song, K., Zhang, D., 2024. Unsupervised
distribution-aware keypoints generation from 3d point clouds. Neural
Networks 173, 106158. URL: https://www.sciencedirect.com/science/
article/pii/S0893608024000820, doi:https://doi.org/10.1016/j.neunet.
2024.106158.
Xu, Q., Tao, W., 2019. Multi-scale geometric consistency guided multi-
view stereo, in: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 5483–5492.
Yao, Y., Luo, Z., Li, S., Fang, T., Quan, L., 2018. Mvsnet: Depth inference
for unstructured multi-view stereo, in: Proceedings of the European
conference on computer vision (ECCV), pp. 767–783.
Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A., 2024. Mip-splatting:
Alias-free 3d gaussian splatting, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 19447–
19456.
Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O., 2018.
The
unreasonable effectiveness of deep features as a perceptual metric, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 586–595.
Zhou, H., Zhao, H., Wang, Q., Hao, G., Lei, L., 2023. Miper-mvs: Multi-
scale iterative probability estimation with refinement for efficient multi-
view stereo. Neural Networks 162, 502–515.
Zwicker, M., Pfister, H., Van Baar, J., Gross, M., 2002. Ewa splatting. IEEE
Transactions on Visualization and Computer Graphics 8, 223–238.
: Preprint submitted to Elsevier
Page 19 of 19
