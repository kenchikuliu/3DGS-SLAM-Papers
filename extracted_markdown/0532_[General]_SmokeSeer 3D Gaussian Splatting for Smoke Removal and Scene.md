<!-- page 1 -->
SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction
Neham Jain, Andrew Jong, Sebastian Scherer, Ioannis Gkioulekas
Carnegie Mellon University
Pittsburgh, PA
nhjain@andrew.cmu.edu, ajong@andrew.cmu.edu, basti@andrew.cmu.edu, igkioule@cs.cmu.edu
Drone 
equipped 
with RGB 
and thermal 
sensors
RGB
Thermal
Desmoked RGB
Insets
Figure 1. Our method utilizes RGB and thermal images from a drone-mounted sensor to perform simultaneous 3D scene reconstruction and
smoke removal using an inverse rendering approach within the 3D Gaussian splatting framework. Insets highlight the effectiveness of our
approach in revealing occluded structures.
Abstract
Smoke in real-world scenes can severely degrade image
quality and hamper visibility. Recent image restoration meth-
ods either rely on data-driven priors that are susceptible to
hallucinations, or are limited to static low-density smoke.
We introduce SmokeSeer, a method for simultaneous 3D
scene reconstruction and smoke removal from multi-view
video sequences. Our method uses thermal and RGB images,
leveraging the reduced scattering in thermal images to see
through smoke. We build upon 3D Gaussian splatting to fuse
information from the two image modalities, and decompose
the scene into smoke and non-smoke components. Unlike
prior work, SmokeSeer handles a broad range of smoke den-
sities and adapts to temporally varying smoke. We validate
our method on synthetic data and a new real-world smoke
dataset with RGB and thermal images. We provide an open-
source implementation and data on the project website.1
1https://imaging.cs.cmu.edu/smokeseer
1. Introduction
Reliable visual perception is essential for safety-critical ap-
plications such as search and rescue, robot navigation, and
industrial inspection. The ability to accurately perceive and
reconstruct 3D environments is particularly vital, as it en-
ables precise spatial reasoning and path planning in complex
scenarios. For example, firefighters navigating through burn-
ing buildings increasingly depend on vision-based systems
to maintain situational awareness. However, dense smoke
severely compromises these systems, obscuring vital envi-
ronment details and increasing operational risks. Develop-
ing technologies that enable these systems to “see through
smoke” is therefore critical for enhancing both safety and
operational effectiveness in these hazardous environments.
Though several approaches have targeted the problem of
enhancing visibility through scattering media, significant
limitations remain. Learning-based approaches that map
hazy to clear images require extensive paired datasets and
typically process individual frames, thereby ignoring valu-
able multi-view constraints. Closer to our work, neural
rendering approaches such as ScatterNeRF [27] and De-
hazeNeRF [4] incorporate physical light transport models
and operate on multi-view RGB data. However, all these
approaches primarily address static haze removal and are
1
arXiv:2509.17329v3  [cs.CV]  23 Dec 2025

<!-- page 2 -->
ill-equipped to handle dense, temporally evolving smoke.
We build an end-to-end system that performs joint 3D
scene reconstruction and smoke removal in the presence
of dense, temporally evolving smoke. Our method uses
images from RGB and thermal cameras, and is effective
on real-world smoke data (Figure 1). We build upon 3D
Gaussian splatting (3DGS) [14] and decompose a smoke-
filled scene into two sets of Gaussians: one representing the
smoke part, and another representing the non-smoke part
of the scene, which we refer to as the surface Gaussians.
This decomposition allows us to render only the surface
Gaussians to visualize the scene without smoke.
Performing this decomposition using only RGB images
is challenging due to the visual ambiguity between light-
reflecting surfaces and light-scattering smoke particles. To
address this challenge, we leverage thermal cameras that
capture long-wavelength infrared radiation, which is substan-
tially less affected by scattering in smoke than visible light.
This property enables thermal sensors to preserve critical
spatial information even in dense smoke conditions. How-
ever, thermal images are low-resolution, have low contrast,
and lack the texture details crucial for object recognition
and scene understanding. Our method overcomes this lim-
itation through a joint optimization strategy that fuses the
robust spatial cues from thermal data with the rich texture
information provided by RGB imagery.
To effectively leverage the complementary strengths of
both modalities, we propose a three-stage approach for
smoke removal and 3D scene reconstruction. In the first
stage, we leverage advances in 3D foundation models [33] to
estimate RGB-thermal poses in the same coordinate frame.
In the second stage, we learn the scene’s geometry exclu-
sively from thermal images, leveraging their robustness in
capturing spatial information even in the presence of dense
smoke. In the third stage, we use both RGB and thermal
images to optimize two sets of Gaussians, for the smoke
and the scene’s surfaces. For the surface Gaussians, we rely
on initialization from the output of the second stage. For
the smoke Gaussians, we use a deformation field to model
the temporal variation of smoke, and enforce handcrafted
priors based on physical properties of smoke. These choices
help ensure that, after optimization, smoke Gaussians exclu-
sively capture the scene smoke, whereas surface Gaussians
accurately represent the underlying scene structure.
Unlike prior learning-based dehazing methods, ours does
not directly rely on image-to-image learned priors and in-
stead formulates smoke removal as an inverse rendering
problem within the 3DGS framework. To the best of our
knowledge, this is the first work that jointly uses RGB and
thermal images for smoke removal and 3D reconstruction.
Our experiments show state-of-the-art results on both
simulated and real-world datasets—collected in partnership
with our county’s fire department using a field operational
drone—for smoke removal and novel view synthesis. Our
code and data are publicly available on the project website,
to ensure reproducibility and facilitate follow-up research.
2. Related Work
2.1. Image-based methods for haze removal
Traditional methods.
Koschmieder [16] developed an at-
mospheric scattering model that describes image formation
under haze as a combination of direct attenuation and airlight.
This model is a simplification of the more general radiative
transfer equation (RTE) [3], which describes the propagation
of light through a medium with scattering and absorption.
Though widely used in dehazing methods, the Koschmieder
model assumes homogeneous static media, limiting its effec-
tiveness for heterogeneous, dynamic smoke conditions.
Early image restoration approaches relied on handcrafted
priors to estimate physical parameters in the Koschmieder
model. He et al. [11] tried to estimate the attenuation map
by leveraging the observation that in most local patches of
haze-free images, at least one color channel has very low
intensity. Zhu et al. [40] proposed the color attenuation prior,
modeling the depth of the scene through the difference be-
tween brightness and saturation. Berman et al. [2] developed
a non-local method based on the observation that colors in
haze-free images form tight clusters in RGB space. Though
effective for thin homogeneous haze, these methods fail in
dense smoke scenarios, for which their priors are ill-suited.
Learning-based methods.
Some recent methods map
hazy to clear images without explicit parameter estimation.
Examples include MSRL-DehazeNet [21], collaborative in-
ference frameworks for dense haze in remote sensing [34],
and saliency-guided mechanisms for UAV imagery [39].
Transformer-based architectures have recently shown
promising results for dehazing. Zamir et al. [37] proposed
Restormer, an efficient transformer for high-resolution image
restoration including dehazing. Guo et al. [10] introduced
a hybrid CNN-transformer architecture that combines local
and global feature extraction. Despite these advances, most
learning-based methods process individual frames indepen-
dently, ignoring valuable temporal and multi-view informa-
tion that could enhance smoke removal performance.
Specific to smoke removal, Salazar-Colores et al. [30]
developed an image-to-image translation approach guided
by an embedded dark channel for desmoking laparoscopy
surgery images. However, this and other similar methods
typically require paired training data (smoke versus smoke-
free), which is challenging to obtain in real-world scenarios,
especially for temporally varying smoke.
2

<!-- page 3 -->
2.2. Neural representations for participating media
Neural radiance Fields (NeRF) [25] have revolutionized
scene representation using continuous volumetric functions.
Several works have extended NeRF to handle participating
media such as smoke and haze. ScatterNeRF [27] incorpo-
rates the Koschmieder model into the NeRF framework, but
remains limited to homogeneous haze conditions. DehazeN-
eRF [4] can handle heterogeneous media but not dynamic
smoke. These methods have primarily focused on static haze
removal and do not address the more challenging problem
of temporally varying smoke—our focus.
3D Gaussian splatting (3DGS) [14] is an efficient alterna-
tive to NeRF through scene representation using 3D Gaus-
sians, enabling real-time rendering. Dynamic 3DGS [24]
extends this framework to dynamic scenes, but does not
specifically address participating media.
Lastly, recent approaches such as ThermalNeRF [19] and
ThermalGaussian splatting [23] incorporate thermal imag-
ing into neural rendering frameworks but do not tackle the
problem of imaging through smoke.
2.3. Multi-modal sensing
Multi-modal sensing has emerged as a promising direction
for robust perception in challenging environments. Thermal
imaging, which captures long-wavelength infrared radiation,
is less affected by smoke and haze compared to RGB cam-
eras [9]. Hwang et al. [12] demonstrated the effectiveness of
fusing RGB and thermal information for object detection in
adverse weather. Li et al. [18] proposed an RGB-thermal ob-
ject tracking benchmark demonstrating the value of thermal
information for robust perception.
Our work bridges these research areas by explicitly mod-
eling temporally varying smoke separately from scene ge-
ometry within the 3DGS framework. Unlike previous ap-
proaches, ours leverages the complementary strengths of
RGB and thermal imaging to achieve 3D reconstruction and
smoke removal without requiring paired training data.
3. Method
We introduce SmokeSeer, a framework for simultaneous
3D scene reconstruction and smoke removal using RGB-
thermal image pairs. Our approach leverages the comple-
mentary strengths of RGB (texture-rich) and thermal (smoke-
penetrating) modalities to address the challenges of dense,
dynamic smoke in safety-critical applications. Our method
comprises three stages (Figure 3): (1) camera pose esti-
mation and smoke segmentation, (2) initial surface recon-
struction from thermal images, and (3) joint optimization of
surface and smoke using both RGB and thermal images.
2
4
6
8
10
12
14
Wavelength ( m)
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
Scattering Coefficient (Qsca)
Scattering Coefficient vs Wavelength
Scattering coefficient (Qsca)
 = 0.55 m (Qsca = 2.48e+00)
 = 8.0 m (Qsca = 3.99e-03)
Figure 2. Scattering coefficient as a function of wavelength (in
µm). We calculate the scattering coefficient using size, refractive
index using standard values for organic matter found in smoke
particles [1]. The scattering coefficient is significantly higher in
the visible spectrum (0.38–0.7 µm) compared to the long-wave
infrared spectrum (8–14 µm). Inset images which are taken from a
drone at roughly the same time illustrate this effect: in visible light
(left), smoke strongly obscures the scene, while in thermal infrared
(right), the underlying structure is clearly visible.
3.1. Use of thermal images
Mie theory [8] provides a framework for understanding
how different types of particles—such as smoke particles—
interact with electromagnetic radiation at different wave-
lengths. For smoke particles of a given size and refractive
index, we can use the Mie theory equations to character-
ize their wavelength-dependent scattering behavior, as illus-
trated in Figure 2. This analysis reveals a crucial insight:
smoke particles predominantly scatter wavelengths in the
visible spectrum (0.38–0.7 µm), where RGB cameras oper-
ate. However, in the long-wave infrared (LWIR) spectrum
(8–14 µm) utilized by thermal cameras, scattering effects
from smoke particles are negligible. This property allows
thermal imaging to penetrate smoke and reveal underlying
surface geometry otherwise obscured in RGB imagery.
In practice, smoke exhibits two key thermal behaviors.
First, smoke is largely transparent in the long-wave infrared
(LWIR) spectrum because heat dissipates rapidly as smoke
moves away from the fire source. This transparency enables
thermal cameras to capture clear views of scene geometry
even when RGB cameras are completely occluded by dense
smoke. However, in regions extremely close to the fire
source, hot smoke can become emissive and appear as a
thermal source rather than remaining transparent, which our
method accounts for in the joint optimization stage.
3.2. Background on 3D Gaussian splatting
Given a collection of posed images {Ik}K
k=1, Ik ∈RH×W
captured from a scene, 3DGS aims to reconstruct a repre-
sentation G of the scene as a set of 3D Gaussians G = {gi}.
3

<!-- page 4 -->
Stage 3
Stage 1
MAST3R
SfM
...
MAST3R
SfM
...
RGB 
Images
Thermal 
Images
Alignment 
Module
Stage 2
(a) RGB poses 
and point cloud
(b) Thermal poses 
and point cloud
(c) RGB/thermal 
poses and point 
cloud in same 
coordinate frame
...
Thermal 
Images
Thermal poses and 
point cloud from 
(c)
Vanilla 3D Gaussian Splatting
Surface 
Gaussians
Smoke 
Gaussians
Render
Render
MLP modeling 
smoke dynamics
Render
Union
Monocular 
depth loss
Priors
Similar 
color/opacity
Priors
Monochromatic
Smoke masks
RGB
Thermal
RGB
Thermal
RGB
Thermal
Coarse 
Surface Reconstruction
Rendering 
loss
Rendering 
loss
Figure 3. An overview of our method, SmokeSeer. The framework consists of three primary stages: (1) Camera pose estimation and smoke
segmentation, (2) Initial surface reconstruction from thermal images, and (3) Joint optimization of surface and smoke plume using both RGB
and thermal images.
Each Gaussian primitive gi is characterized by a center posi-
tion µi, a symmetric positive-definite covariance matrix Ωi,
an alpha value αi, and appearance attributes encoded using
spherical harmonic coefficients hi [26]. Unlike approaches
requiring different representations for surfaces (e.g., meshes
or implicits) and volumes (e.g., voxel grids), Gaussian prim-
itives can represent both surfaces and smoke, simplifying
optimization and rendering.
3.3. Modeling scattering media using Gaussians
We decompose the smoke-filled scene into two sets of Gaus-
sians: surface Gaussians G representing surfaces in the scene,
and smoke Gaussians S capturing the dynamic smoke plume.
Before detailing these sets, we explain how to render images
using Gaussian primitives.
We first define the transmittance function, which is central
to volumetric rendering. For a ray r(t) = o + td starting at
position o in direction d, the transmittance Tσ(t) represents
the probability that the ray travels from its origin to point
r(t) without obstruction. It is defined as:
Tσ(t) := exp

−
Z t
tn
σ(s)ds

,
(1)
where σ(s) is the density function along the ray, and tn is
the near-plane distance. In a scene with both surfaces and
smoke, we have two density functions: σ(t) for surfaces and
σs(t) for smoke. The combined transmittance is:
Tσ+σs(t) = exp

−
Z t
tn
[σ(s) + σs(s)] ds

(2)
= Tσ(t) · Tσs(t),
(3)
which represents the probability of the ray reaching point
r(t) without hitting either a surface or smoke particles.
Chen et al. [4] have shown that the volume rendering
equation for a scene with mixed density takes the form:
C(r, d) =
Z t0
tn
c(r(t), d)σ(t)Tσ+σs(t) dt
|
{z
}
Csurface
+
Z t0
tn
cs(r(t))σs(t)Tσ+σs(t) dt
|
{z
}
Csmoke
,
(4)
where t0 is the far-plane distance. Our dual Gaussian repre-
sentation directly maps to this equation, where the surface
Gaussians G correspond to Csurface with color c and opacity
σ, whereas the smoke Gaussians S correspond to Csmoke
with color cs and opacity σs. Rendering the union G ∪S is
equivalent to computing the rendering equation (4).
The rendering equation for the clear-view surfaces with-
out smoke interference is given by:
Cclear(r, d) =
Z t0
tn
c(r(t), d)σ(t)Tσ(t) dt.
(5)
Rendering only the surface Gaussians G is equivalent to
computing the rendering equation (5). By modeling surface
and smoke separately, we achieve effective smoke removal
through selectively rendering only surface Gaussians.
3.4. Modality-specific representations
Building on the Gaussian representation described in Sec-
tion 3.2, we extend the model to handle both RGB and ther-
mal modalities. We use {IRGB
k
}KRGB
k=1 and {IT
k}KT
k=1 to denote
our RGB and thermal image collections respectively, with
associated camera poses P RGB
k
and P T
k .
For our dual Gaussian representation:
• Surface Gaussians G maintain the parameters from Sec-
tion 3.2 but with modality-specific spherical harmonic
coefficients (hRGB
i
, hT
i ), and modality-shared opacity α.
4

<!-- page 5 -->
• Smoke Gaussians S have modality-specific spherical har-
monic coefficients and opacities (αRGB
i
≫αT
i ), reflecting
the physical properties described in Section 3.1. In addi-
tion, they are time-varying to capture smoke dynamics.
3.5. Stage 1: Generating segmentation masks and
obtaining poses
In this stage, our objective is to estimate camera poses for
RGB and thermal images in a common coordinate system.
Accurate cross-modal poses are a prerequisite for any multi-
view fusion; without them, correspondence and consistency
losses are ill-defined. This task is challenging due to the
different sensor responses between these modalities, which
complicates cross-modal feature matching. Additionally, the
featureless appearance and dynamic smoke in RGB images
impede reliable feature extraction.
We address these challenges with a three-step approach:
1. Smoke segmentation: We use GroundedSAM [29], based
on SAMv2 [28], to identify and mask out smoke-affected
regions in RGB images. Doing so ensures we match only
features from reliable, smoke-free areas.
2. Independent 3D reconstructions: We run MAST3R-
SfM [7] independently on RGB and thermal images. Us-
ing the masks from the previous step, we discard matches
in the smoke regions of RGB images. Though MAST3R-
SfM handles RGB-RGB and thermal-thermal matching
well, it struggles with RGB-thermal matching.
3. Cross-modal registration: We use MINIMA [13], which
is specialized for cross-modality matching, to establish
2D correspondences between RGB-thermal image pairs.
We then lift these correspondences to 3D using the 2D-3D
mappings from the per-modality calibration. Doing so en-
ables the estimation of a similarity transform T ∈Sim(3)
that aligns the RGB and thermal coordinate systems.
3.6. Stage 2: Reconstructing the scene using ther-
mal images
In this stage, we obtain a first reconstruction of the scene
geometry using only thermal images, which are minimally af-
fected by smoke. We run vanilla 3D Gaussian splatting [14]
on the thermal sequence, which outputs a smoke-free repre-
sentation of the scene geometry. The surface reconstruction
is coarse due to the low resolution of thermal images, but
serves as a reliable initialization for our surface Gaussians.
3.7. Stage 3: Fusing RGB-thermal information and
refining geometry
In the final stage, we jointly optimize surface and smoke
Gaussian sets using both RGB and thermal images:
• Surface Gaussians: Initialized from Stage 2, these Gaus-
sians remain static and maintain identical opacity across
modalities. We augment them with spherical harmonic
coefficients to capture RGB appearance.
• Smoke Gaussians: Randomly initialized within the scene
bounds, these Gaussians evolve temporally and exhibit
modality-dependent opacity, to model smoke’s varying
visibility in RGB versus thermal images (Section 3.1).
Though in principle we could use Mie theory to model
opacities, we opt for a more flexible approach with two in-
dependent variables for smoke visibility in each modality.
3.7.1
Modeling the dynamic smoke
Our approach explicitly accounts for the temporal evolution
of smoke, which is critical for applications such as firefight-
ing where smoke behavior is dynamic and unpredictable.
Accounting for smoke motion enables more accurate sur-
face reconstruction in areas temporarily occluded by passing
smoke, and improves separation of surface and smoke. We
model the dynamics of smoke following the deformable 3D
Gaussians framework [36]. This framework uses 3D Gaus-
sians in a canonical space, along with a deformation field to
model motion over time. To model this field, we use a multi-
layer perceptron (MLP) that takes as input the positions of
the 3D Gaussians and a timestep t, and outputs offsets in
position, scale, and rotation. These offsets transform the
canonical 3D Gaussians to the deformed space at each time.
We use a bimodal Gaussian distribution following [17] to
model smoke opacity as a function of time.
3.7.2
Priors on properties of smoke Gaussians
To facilitate accurate surface-smoke separation and model-
ing of realistic smoke behavior, during optimization we use
priors motivated by physical properties of smoke:
• Smoke consistency: We minimize variance in opacity and
color across smoke Gaussians:
Lsmoke alpha = Var({αi}i∈S)
(6)
Lsmoke color = Var({ci}i∈S)
(7)
This prior reflects the physical observation that smoke
particles in a local region typically have similar optical
properties. In real smoke, particles of similar size and
composition would have nearly identical opacity and scat-
tering properties. By enforcing consistency across smoke
Gaussians, we prevent unrealistic variations from arising
during optimization. Though the loss should ideally ap-
ply to Gaussians in local neighborhoods, we found that
applying it across all Gaussians works well in practice.
• Monochromaticity: We enforce consistent color channels
across smoke Gaussians:
Lmono =
X
i∈S
Var(cR
i , cG
i , cB
i ).
(8)
This prior reflects the physical property that smoke typi-
cally appears as a neutral gray color. It prevents our model
from generating implausible colored smoke.
5

<!-- page 6 -->
• Depth consistency: We align the surface Gaussians with
monocular depth cues:
Ldepth = ∥di −ˆdi∥,
(9)
where di denotes predicted depth on a thermal image using
a monocular depth estimation model [35] and ˆdi is the ren-
dered depth from the surface Gaussians using thermal cam-
era parameters. This prior leverages the smoke-penetrating
property of thermal imaging (Section 3.1). Since thermal
images are minimally affected by smoke, they provide
reliable depth cues for the underlying surface geometry,
helping to prevent surface Gaussians from being incor-
rectly positioned in smoke-occluded regions.
• Mask alignment: The alpha values of smoke Gaussians
should be consistent with the masks from Stage 1:
Lmask = ∥Mpred −MGT∥1,
(10)
where Mpred and MGT are the pixel-wise accumulated
alpha values of the rendered smoke Gaussians and seg-
mentation masks, respectively. This prior ensures spatial
consistency between our reconstructed smoke volume and
the observed smoke regions in input images. It helps
constrain the optimization to place smoke Gaussians in
only regions with smoke present, and prevent them from
appearing in smoke-free ones.
The total optimization loss is a weighted sum of these
physically-motivated priors and a standard rendering loss for
RGB and thermal images:
Ltotal = λrenderLrender + λsmoke alphaLsmoke alpha
+ λsmoke colorLsmoke color + λmonoLmono
+ λdepthLdepth + λmaskLmask.
(11)
This formulation enables separation of scene geometry
from smoke, while maintaining physical consistency across
the RGB and thermal modalities.
4. Experimental evaluation
We evaluate our method on synthetic and real-world datasets
to demonstrate its effectiveness for smoke removal and 3D
scene reconstruction. We compare against state-of-the-art
methods, and validate our design choices through ablation
studies. We provide implementation details in the supple-
ment, and video results on the project website.
4.1. Datasets
Synthetic dataset.
For quantitative evaluation with ground
truth, we create a synthetic dataset using Blender’s
Mantaflow [32] smoke simulator. The dataset comprises
10 scenes: 5 object-level scenes from the NeRF synthetic
dataset [25], and 5 large-scale scenes. For each scene, we
generate 150 RGB and thermal frames with dynamic smoke.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
ImgDehaze + 3DGS
14.42
0.37
0.318
Ours (RGB only)
15.08
0.38
0.326
Ours (Full)
19.92
0.76
0.247
Table 1. Quantitative results on the synthetic dataset for novel view
synthesis. Our full method outperforms all baselines.
Real-world dataset.
In collaboration with our county’s
fire department, we collected a real-world dataset using a
Spirit drone equipped with roughly co-located RGB and ther-
mal cameras. There is no time synchronization between the
frames captured by the RGB and thermal cameras on the
drone, which makes relative pose estimation challenging. We
do not report quantitative metrics for the real-world dataset
as obtaining true ground truth is challenging in such environ-
ments. Instead, we provide an approximation which we refer
to as “Reference” in the figures. We provide more details
in the supplement. This dataset presents several challenges
not found in synthetic data, including: imperfect alignment
between RGB and thermal cameras, unpredictable smoke
motion due to wind, and motion blur from drone movement.
These factors make our real-world dataset a rigorous bench-
mark for evaluating the practical utility of smoke removal
algorithms in safety-critical applications.
4.2. Baseline methods
We compare three methods:
• ImgDehaze + 3DGS: A two-stage approach that first
applies a state-of-the-art single-image dehazing method
(ConvIR [6]) to each RGB frame, then uses deformable
3DGS [36] on the dehazed images.
• Ours (RGB only): Our approach using only RGB images
(Stage 3 without thermal input).
• Ours (Full): Our complete approach using both RGB and
thermal images.
We could not compare with DehazeNeRF [4], the prior work
closest to ours, due to lack of open-source code.
4.3. Results
Synthetic data results.
Table 1 presents quantitative re-
sults for novel view synthesis on our synthetic dataset, using
the PSNR, SSIM, and LPIPS [38] metrics. Our full method
outperforms all baselines across all metrics, especially in
scenes with heavy smoke where we achieve a PSNR gain of
up to 4.8 dB over the RGB-only approaches.
Figure 4 shows qualitative results on the synthetic dataset.
Our method removes smoke while preserving fine details
in the scene. In contrast, baseline methods either fail to
completely remove smoke or introduce artifacts.
Real-world data results.
Figure 5 demonstrates our
method’s effectiveness on real-world data. The improve-
6

<!-- page 7 -->
Thermal Image
RGB Image
Ground Truth
Ours (Full)
ImgDehaze
Ours (RGB only)
Figure 4. Qualitative results on the synthetic dataset. Our full method effectively removes smoke while preserving structural and texture
details, outperforming RGB-only approaches.
Thermal Image
RGB Image
Reference
Ours (Full)
ImgDehaze
Ours (RGB only)
Figure 5. Qualitative results on the real-world dataset. Our method successfully removes smoke in challenging real-world conditions while
preserving scene details.
ment is particularly noticeable in regions with dense smoke,
where baseline methods struggle to reconstruct the scene
geometry. Our method fuses complementary information
in RGB and thermal modalities, to improve smoke removal
while preserving texture details critical for scene understand-
ing.
Though real-world reconstructions exhibit some artifacts,
e.g., residual wisps of smoke or reduced color saturation un-
7

<!-- page 8 -->
w/o Lsmoke alpha
w/o Lsmoke color
w/o Lmono
w/o Ldepth
w/o Lmask
Full method
Figure 6. Visual comparison of ablation configurations. Each image shows the result of removing a specific prior from our full method. Our
method is able to recover the bricks in the wall of the house better than other configurations.
Configuration
PSNR ↑
SSIM ↑
LPIPS ↓
w/o Lsmoke alpha
18.96
0.68
0.312
w/o Lsmoke color
19.02
0.71
0.289
w/o Lmono
19.78
0.76
0.248
w/o Ldepth
19.88
0.75
0.252
w/o Lmask
19.82
0.74
0.251
Ours (Full)
19.92
0.76
0.247
Table 2. Ablation study showing the impact of each component in
our framework. Each prior contributes to the overall performance,
with the depth consistency prior having the most significant impact.
der extreme occlusion (Figure 5), they represent a substantial
improvement in situational awareness. For first responders,
the ability to discern room layout, locate doorways, and
identify obstacles even at reduced fidelity turns an unusable,
smoke-obscured video stream into an actionable 3D map.
4.4. Ablation study
Table 2 shows an ablation study on individual components
in our framework. Each component provides a measurable
performance improvement, and the combination of all com-
ponents yields the best results. The depth consistency prior
(Ldepth) significantly improves performance, highlighting
the importance of leveraging thermal information for accu-
rate geometry reconstruction in smoke-filled environments.
Figure 6 provides a visual comparison of different ab-
lation configurations. Without the smoke consistency pri-
ors (Lsmoke alpha and Lsmoke color), the model struggles to
separate smoke from surfaces. Without the monochromatic-
ity prior (Lmono), the model generates unrealistic colored
smoke. The depth consistency prior (Ldepth) is important
for real-world scenes where the camera poses might be noisy.
The mask alignment prior (Lmask) helps localize smoke and
place smoke Gaussians in the correct location.
5. Conclusion
We presented SmokeSeer, a framework for joint 3D scene
reconstruction and smoke removal in dynamic smoke-filled
environments. Our key insight is to leverage the complemen-
tary strengths of RGB and thermal imaging to decompose the
scene into its surface and smoke components. We achieved
this using a 3DGS-based inverse rendering pipeline to opti-
mize separate smoke and surface Gaussians, appropriately
regularized to account for their different physical properties.
We demonstrated our method on synthetic datasets and real-
world environments representative of firefighting settings.
Our experiments in real-world firefighting scenarios demon-
strate practical viability for emergency response applications.
By publicly releasing our code and dataset, we aim to estab-
lish a foundation for future research in vision through smoke
and multimodal scene understanding.
Limitations and future work.
Though our method
achieves state-of-the-art performance in smoke removal and
3D reconstruction, several limitations remain. First, we
model the temporal evolution of smoke using a deformation
field, without explicitly incorporating physics-based priors
such as fluid dynamics. Future work could integrate pri-
ors based on the Navier-Stokes equations to better capture
smoke’s physical behavior [5]. Second, our method requires
careful balancing of multiple loss terms during optimization
(Equation 11), and cannot handle very dense smoke that
might occlude the scene completely. Future work could in-
corporate generative priors to provide stronger guidance at
regions heavily occluded by smoke.
Acknowledgments.
We thank Ian Higgins and John Keller
for help with flying the drone during data acquisition;
Sreekar Ranganathan for help with the camera hardware
and data collection setup; and Jeff Tan and Nikhil Keetha
for helpful discussions. This work was supported by Na-
tional Institute of Food and Agriculture award 2023-67021-
39073; Defense Science and Technology Agency contract
#DST000EC124000205; and Alfred P. Sloan Research Fel-
lowship FG202013153 for Ioannis Gkioulekas. This work
used Bridges-2 at the Pittsburgh Supercomputing Center,
through allocation cis220039p from the Advanced Cyberin-
frastructure Coordination Ecosystem: Services & Support
(ACCESS) program, which is supported by National Science
Foundation awards 2138259, 2138286, 2138307, 2137603,
and 2138296; and National Artificial Intelligence Research
Resource Pilot-2211.
References
[1] E. Alonso-Blanco, A. I. Calvo, R. Fraile, and A. Castro. The
influence of wildfires on aerosol size distributions in rural
areas. The Scientific World Journal, 2012:735697, 2012. 3
8

<!-- page 9 -->
[2] Dana Berman, Tali Treibitz, and Shai Avidan. Non-local
image dehazing. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 1674–1682,
2016. 2
[3] Subrahmanyan Chandrasekhar. Radiative Transfer. Dover
Publications, New York, 1960. Unabridged and slightly re-
vised version of the work first published in 1950. 2
[4] W. Chen, W. Yifan, S. Kuo, and G. Wetzstein. Dehazenerf:
Multiple image haze removal and 3d shape reconstruction
using neural radiance fields. In 3DV, 2024. 1, 3, 4, 6
[5] Mengyu Chu, Lingjie Liu, Quan Zheng, Erik Franz, Hans-
Peter Seidel, Christian Theobalt, and Rhaleb Zayer. Physics
informed neural fields for smoke reconstruction with sparse
data. ACM Trans. Graph., 41(4), 2022. 8
[6] Yuning Cui, Wenqi Ren, Xiaochun Cao, and Alois Knoll. Re-
vitalizing convolutional network for image restoration. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
2024. 6
[7] Bardienus Pieter Duisterhof, Lojze ˇZust, Philippe Weinza-
epfel, Vincent Leroy, Yohann Cabon, and J´erˆome Revaud.
Mast3r-sfm: a fully-integrated solution for unconstrained
structure-from-motion. ArXiv, abs/2409.19152, 2024. 5, 11
[8] Jeppe Revall Frisvad, Niels Jørgen Christensen, and Hen-
rik Wann Jensen. Computing the scattering properties of
participating media using lorenz-mie theory. In ACM SIG-
GRAPH 2007 Papers, page 60, New York, NY, USA, 2007.
Association for Computing Machinery. 3
[9] Rikke Gade and Thomas B. Moeslund. Thermal cameras and
applications: A survey. Machine Vision and Applications, 25
(1):245–262, 2014. 3
[10] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy,
Junhui Hou, Sam Kwong, and Runmin Cong. Image dehazing
transformer with transmission-aware 3d position embedding.
IEEE Transactions on Circuits and Systems for Video Tech-
nology, 33(1):25–39, 2022. 2
[11] Kaiming He, Jian Sun, and Xiaoou Tang. Single image haze
removal using dark channel prior. In IEEE Transactions on
Pattern Analysis and Machine Intelligence, pages 2341–2353.
IEEE, 2011. 2
[12] Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and
In So Kweon. Multispectral pedestrian detection: Benchmark
dataset and baseline. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 1037–
1045, 2015. 3
[13] Xingyu Jiang, Jiangwei Ren, Zizhuo Li, Xin Zhou, Dingkang
Liang, and Xiang Bai. Minima: Modality invariant image
matching. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2025. 5, 11, 12
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Transactions on Graphics, 42(4), 2023.
2, 3, 5
[15] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. In 3rd International Conference on
Learning Representations, ICLR 2015, San Diego, CA, USA,
May 7-9, 2015, Conference Track Proceedings, 2015. 11
[16] H. Koschmieder. Theorie der horizontalen Sichtweite. Keim
& Nemnich, 1924. 2
[17] Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae, and
Hae-Gon Jeon. Fully explicit dynamic gaussian splatting. In
Proceedings of the Neural Information Processing Systems,
2024. 5
[18] Chenglong Li, Xinyan Liang, Yijuan Lu, Nan Zhao, and Jin
Tang. Rgb-t object tracking: Benchmark and baseline. Pattern
Recognition, 96:106977, 2019. 3
[19] Yvette Y Lin, Xin-Yi Pan, Sara Fridovich-Keil, and Gor-
don Wetzstein. ThermalNeRF: Thermal radiance fields. In
IEEE International Conference on Computational Photogra-
phy (ICCP). IEEE, 2024. 3
[20] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun
Zhu, et al. Grounding dino: Marrying dino with grounded
pre-training for open-set object detection. arXiv preprint
arXiv:2303.05499, 2023. 11
[21] Xiaohong Liu, Yongrui Ma, Zhihao Shi, and Jun Chen. Multi-
scale residual learning for single image dehazing. In IEEE
International Conference on Multimedia and Expo (ICME),
pages 1366–1371. IEEE, 2019. 2
[22] David G Lowe. Object recognition from local scale-invariant
features. In Proceedings of the seventh IEEE international
conference on computer vision, pages 1150–1157. Ieee, 1999.
11
[23] Rongfeng Lu, Hangyu Chen, Zunjie Zhu, Yuhang Qin, Ming
Lu, Le Zhang, Chenggang Yan, and Anke Xue. Thermalgaus-
sian: Thermal 3d gaussian splatting, 2024. 3
[24] Jonathon Luiten, Vincent Leroy, Julian Ost, Fabian Manhardt,
Francis Engelmann, Deva Ramanan, and Federico Tombari.
Dynamic 3d gaussians: Tracking by persistent dynamic view
synthesis. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 22398–22408, 2023.
3
[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view synthe-
sis. In ECCV, 2020. 3, 6
[26] Ravi Ramamoorthi and Pat Hanrahan. An efficient repre-
sentation for irradiance environment maps. In Proceedings
of the 28th Annual Conference on Computer Graphics and
Interactive Techniques, page 497–500, New York, NY, USA,
2001. Association for Computing Machinery. 4
[27] Andrea Ramazzina, Mario Bijelic, Stefanie Walz, Alessan-
dro Sanvito, Dominik Scheuble, and Felix Heide. Scattern-
erf: Seeing through fog with physically-based inverse neural
rendering. In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pages 17957–17968,
2023. 1, 3
[28] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting
Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan
Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer.
Sam 2: Segment anything in images and videos, 2024. 5, 11
[29] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li,
He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan,
Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang
9

<!-- page 10 -->
Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling
open-world models for diverse visual tasks, 2024. 5, 11
[30] Sebasti´an Salazar-Colores, Hugo M. Jim´enez, C´esar J. Ortiz-
Echeverri, and Gerardo Flores.
Desmoking laparoscopy
surgery images using an image-to-image translation guided by
an embedded dark channel. IEEE Access, 8:208898–208909,
2020. 2
[31] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In Proceedings of the IEEE confer-
ence on computer vision and pattern recognition, pages 4104–
4113, 2016. 11
[32] Nils Thuerey and Tobias Pfaff.
MantaFlow,
2018.
http://mantaflow.com. 6
[33] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud.
Dust3r: Geometric 3d
vision made easy. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 20697–20709, 2024. 2
[34] Wenjing Wang, Yuan Yuan, Qi Wu, Xiangyu Li, and Yanyun
Zhang. Dynamic collaborative inference for dense haze re-
moval in remote sensing imagery. IEEE Transactions on
Geoscience and Remote Sensing, 61:1–15, 2023. 2
[35] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything
v2, 2024. 6
[36] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin.
Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. arXiv
preprint arXiv:2309.13101, 2023. 5, 6, 11
[37] Syed Waqas Zamir, Aditya Arora, Salman Khan, Mu-
nawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang.
Restormer: Efficient transformer for high-resolution image
restoration. In CVPR, pages 5728–5739, 2022. 2
[38] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 6
[39] Xiaolong Zhao, Yingjie Jiang, Weilong Ding, Feng Huang,
and Wenbing Tao. Saliency-guided image dehazing for uav
imagery.
IEEE Transactions on Geoscience and Remote
Sensing, 62:1–15, 2024. 2
[40] Qingsong Zhu, Jiaming Mai, and Ling Shao. Fast single
image haze removal algorithm using color attenuation prior.
IEEE Transactions on Image Processing, 24(11):3522–3533,
2015. 2
10

<!-- page 11 -->
SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction
Supplementary Material
Input RGB
RGB with Smoke Mask
Figure 7. Output of smoke segmentation pipeline. From left to right:
Input RGB images with smoke, generated smoke masks. Note how
the segmentation model accurately identifies smoke regions even
with varying density and illumination.
This supplement provides additional visualizations and
implementation details that complement the main paper.
6. Smoke Segmentation Results
Grounded-SAM [29] integrates Grounding DINO [20], an
open-set object detector, with the Segment Anything Model
(SAMv2) [28], to facilitate text-driven object detection and
segmentation. We use this framework to automatically gen-
erate segmentation masks for smoke by inputting the prompt
“smoke.” These masks are crucial for reliable feature match-
ing in Stage 1 of our pipeline, and also serve as supervision
for the mask alignment loss in Stage 3. Figure 7 shows
example smoke segmentation masks.
7. Feature Matching Comparison
Figure 8 compares traditional SIFT matching [22] with
MAST3R-SfM [7] for feature matching in smoke-affected
scenes. The comparison highlights how SIFT matching
fails for low-texture thermal images, while MAST3R-SfM
with smoke masking provides more reliable correspondences.
This robust matching is essential for the accurate camera
pose estimation required in Stage 1 of our pipeline.
8. Cross-Modal Registration Details
Figure 9 visualizes the cross-modal registration process us-
ing MINIMA [13] for aligning RGB and thermal coordinate
systems. We find that running COLMAP [31] on our data
fails to register the images and gives a degenerate result.
Figure 8. Feature matching comparison between images of the
same modality (thermal). Top: SIFT feature matching fails in
low-texture, low-contrast thermal images. Bottom: MAST3R-
SfM provides more reliable correspondences by leveraging learned
features that are more robust to the challenges of thermal imagery.
Running MAST3R-SfM [7] does give better results than
COLMAP and is able to register all images. However, it is
still not perfect and there is some misalignment. The visual-
ization shows the 2D correspondences established between
RGB and thermal image pairs and the resulting aligned point
clouds. This cross-modal registration is crucial for our ap-
proach, as it allows us to get the RGB and thermal images in
the same coordinate system which is necessary for running
the subsequent 3D reconstruction pipeline.
9. Implementation Details
We provide implementation details to facilitate reproducibil-
ity. We implemented our framework in PyTorch, building on
the official 3DGS repository. For deformable Gaussian splat-
ting, we adapted the implementation from Yang et al. [36].
We used the default parameters and configurations for 3DGS
and Deformable 3DGS. We used the same hyperparame-
ters for both synthetic and real-world experiments. We
trained Stages 2 and 3 for 15000 iterations each, using the
Adam optimizer [15]. For Stage 3, we set λrender = 1.0,
λsmoke alpha = 0.1, λsmoke color = 0.05, λmono = 0.1,
λdepth = 2.0, and λmask = 0.5. We ran all experiments
on a workstation with an NVIDIA RTX 4090 GPU, an Intel
i9-13900K CPU, and 128 GB RAM. For a typical scene in
our real-world dataset, our method takes 1 hour for Stage 1,
10 minutes for Stage 2 and 30 minutes for Stage 3.
11

<!-- page 12 -->
Figure 9. Cross-modal feature matching comparison. Top: Tra-
ditional SIFT feature matching fails between RGB and thermal
images due to fundamental differences in appearance across modal-
ities. Bottom: MINIMA [13] provides more reliable correspon-
dences by explicitly addressing cross-modal challenges, enabling
accurate registration of the two sensor types.
10. Reference for real-world experiments
We do not report quantitative metrics for the real-world
dataset as obtaining true ground truth is challenging in such
environments. Instead, we provide an approximation that we
refer to as “Reference” in figures. To create this reference,
we collected additional smoke-free RGB images of the same
scenes in a separate drone flight. We then: (1) reconstructed
the smoke-free scene using 3DGS, (2) obtained the cam-
era poses of smoke-filled and smoke-free image sets in the
same coordinate frame, and (3) rendered novel views of the
smoke-free reconstruction using the camera poses from the
smoke-filled sequence. These reference images serve as an
approximate benchmark, though they are not perfect ground
truth as the poses are noisy and environmental conditions
may have changed between captures.
12
