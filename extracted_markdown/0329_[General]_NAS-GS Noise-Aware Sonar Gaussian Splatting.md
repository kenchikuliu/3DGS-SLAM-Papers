<!-- page 1 -->
NAS-GS: Noise-Aware Sonar Gaussian Splatting
Shida Xu1, Jingqi Jiang1, Jonatan Scharff Willners2, and Sen Wang1
Abstract— Underwater sonar imaging plays a crucial role
in various applications, including autonomous navigation in
murky water, marine archaeology, and environmental monitor-
ing. However, the unique characteristics of sonar images, such
as complex noise patterns and the lack of elevation information,
pose significant challenges for 3D reconstruction and novel
view synthesis. In this paper, we present NAS-GS, a novel
Noise-Aware Sonar Gaussian Splatting framework specifically
designed to address these challenges. Our approach introduces
a Two-Ways Splatting technique that accurately models the
dual directions for intensity accumulation and transmittance
calculation inherent in sonar imaging, significantly improving
rendering speed without sacrificing quality. Moreover, we pro-
pose a Gaussian Mixture Model (GMM) based noise model
that captures complex sonar noise patterns, including side-
lobes, speckle, and multi-path noise. This model enhances the
realism of synthesized images while preventing 3D Gaussian
overfitting to noise, thereby improving reconstruction accuracy.
We demonstrate state-of-the-art performance on both simulated
and real-world large-scale offshore sonar scenarios, achieving
superior results in novel view synthesis and 3D reconstruction.
I. INTRODUCTION
Underwater inspections are vital for maintaining offshore
infrastructure, conducting marine archaeology, and monitor-
ing the environment. While optical cameras are effective in
clear waters, imaging sonar systems offer robust perception
in turbid environments where light penetration is limited.
However, the distinct nature of acoustic imaging character-
ized by complex noise patterns, low resolution, and nonlinear
geometry presents substantial hurdles for 3D reconstruction
and novel view synthesis.
Recent progress in differentiable rendering, such as Neural
Radiance Fields (NeRF) and Gaussian Splatting (GS) [1], [2],
[3], [4], [5], has enabled photorealistic view synthesis and
geometric reconstruction for optical cameras. Yet, directly
applying these methods to sonar imaging is ineffective due to
fundamental differences in sensing modalities. Sonar images
possess unique properties, including intensity accumulation
along elevation arcs, range-dependent transmittance, and
significant noise artifacts like side-lobes, speckle, and multi-
path reflections. These factors demand specialized geometric
modeling and robust noise handling strategies.
Differentiable rendering for sonar has recently gained
traction. Early works utilized NeRF and signed distance
functions (SDF) for sonar synthesis and reconstruction. Neu-
sis [6], [7] proposed a physics-based sonar rendering model
within a NeRF framework, facilitating novel view synthesis
1
I-X and Department of Electrical and Electronic Engineering,
Imperial
College
London,
UK
{s.xu23, j.jiang23,
sen.wang}@imperial.ac.uk
2Frontier Robotics, The National Robotarium, Edinburgh UK
Fig. 1: Overview of the proposed NAS-GS pipeline.
and 3D reconstruction. Subsequent work [8] extended this
to handle pose drift. However, NeRF-based approaches are
often computationally intensive and slow, limiting real-time
utility. Furthermore, they frequently struggle to model com-
plex sonar noise, resulting in reduced reconstruction quality
on real-world scenarios with noisy data. Differentiable Space
Carving (DSC) [9] offers a more efficient alternative but
lacks the ability to model complex noise characteristics. As
noted in [10], both NeRF-based methods and DSC suffer
from limited quality in novel view synthesis.
Gaussian Splatting offers faster rendering speeds than
NeRF due to its explicit 3D representation, making it suitable
for real-time applications. Recent efforts have adapted GS
to sonar and fuse with camera data. ZSplat [11] uses a
simplified sonar geometry model, which limits its efficacy
in real-world scenarios. Aqua-Splat [12] incorporates a more
accurate physics-based forward model but lacks explicit
noise modeling and relies on computationally expensive
volume rendering. This approach, while adequate for small,
controlled environments, may be inefficient for large-scale
offshore scenes. Both methods also exhibit performance
degradation without camera input due to insufficient noise
modeling and have primarily been validated on simulated or
small-scale tank data.
SonarSplat [10] is the most relevant prior work, focusing
on sonar-based Gaussian Splatting. Although it introduces
a sonar-specific pipeline and a noise model for azimuth
streaking, it has notable limitations: (1) its noise model sim-
plifies noise as per-pixel gain, which handles dark azimuth
streaking but fails to capture complex side-lobe, speckle, and
multi-path noise; and (2) its rendering speed is considerably
slower than camera-based methods. These issues hinder
its application in large-scale offshore environments where
complex noise is prevalent and efficiency is paramount.
arXiv:2601.06285v1  [cs.CV]  9 Jan 2026

<!-- page 2 -->
To overcome these limitations, we propose NAS-GS, a
novel Noise-Aware Sonar Gaussian Splatting framework as
shown in Figure 1, designed for accurate sonar imaging
through rigorous geometric modeling and comprehensive
noise characterization. Our main contributions are:
• A novel Two-Ways Splatting technique that accurately
models the unique projection characteristics of imaging
sonar. It efficiently manages the dual directions of
intensity accumulation and transmittance calculation,
boosting rendering speed to over 700 Frames Per Sec-
ond (FPS) without compromising quality.
• A learnable Gaussian Mixture Model (GMM) based
noise model that captures complex sonar noise patterns,
including side-lobes, speckle, and multi-path noise. This
model improves the realism of synthesized images and
prevents 3D Gaussians from overfitting to noise, thereby
enhancing reconstruction accuracy. Furthermore, our
framework supports rendering both realistic noisy im-
ages and denoised images for simulation and data
augmentation.
• We demonstrate state-of-the-art performance on both
simulated and real-world large-scale offshore sonar
datasets, achieving superior results in novel view syn-
thesis and 3D reconstruction.
We will release our code, simulation dataset upon the
paper’s acceptance. For more details, please visit our project
page1.
II. THE PROPOSED NAS-GS METHOD
A. Sonar Geometry
To formulate sonar geometry, we define multiple coordi-
nate frames: the world frame (W), the sonar Cartesian frame
(S), an intermediate 3D representation sonar polar-elevation
(SPE), and two intermediate 2D representations sonar polar
(SP) and sonar elevation-azimuth (SE) which are subsequently
mapped to their respective image pixel coordinates (IP and
IE). The transformation pipeline consists of a rigid body
transformation from world to sonar Cartesian coordinates,
followed by a non-linear transformation to polar-elevation
coordinates, then projection functions that convert 3D polar-
elevation coordinates into 2D representations, and finally
similarity transformations that map these representations to
pixel coordinates. The Sonar Geometry model is shown in
Figure 2.
1) Coordinate Frame Definitions: A point in the world
frame and the sonar frame are denoted as:
Wpi .=


x
y
z


and
Spi .=


xs
ys
zs


(1)
2) World to Sonar Transformation: The transformation
from the world frame to the sonar frame is given by the
homogeneous transformation matrix:
TSW .=

RSW
StSW
0
1

(2)
1https://senseroboticslab.github.io/NAS-GS-Page/
Azimuth
Elevation
3D Cartesian to 
Polar-Elevation Frame
3D Cartesian Frame
3D Polar-Elevation Frame
Range
S
φmax
φmax
θmax
SPE
x
W
TSW
Spi
φi
φi
φmax
rmax
rmax
θmax θi
θi
ri
ri
y
z
Spi
(a) Sonar coordinate frames and transformations
Image X
Image Y
SPE
φi
θi
ri
Spi
SPE
φi
θi
ri
Spi
3D Polar-Elevation Frame
Projection
Projection
2D Polar Frame
Polar Image Frame
2D Azimuth-Elevation Frame
Azimuth-Elevation Image Frame
SE
SP
IP
SIP SP
SIE SE
φi
IPθi
θi
θi
IPri
ri
3D Polar-Elevation Frame
IE
IEθi
IEφi
(b) Sonar projection from 3D to 2D image space
Fig. 2: Sonar Geometry Model
where RSW is the rotation matrix and StSW is the translation
vector. A point in the sonar frame is obtained by:
Spi = TSW Wpi
(3)
3) Cartesian to Polar-Elevation Conversion: A 3D point
in the sonar frame can be projected into a 3D polar-elevation
coordinate system, whose x and y stand for the azimuth angle
and range respectively, and z stands for the elevation angle.
The projection process is first to shift the origin along the z
axis, then rotate the elevation-range planes of both side as
shown in figure 2b. Therefore, the function πpe that maps
a 3D Cartesian point in the sonar frame to polar-elevation
coordinates is
SPEpi = πpe( Spi)
=


ri
θi
ϕi

=


p
x2s + y2s + z2s
atan2(ys, xs)
arcsin

zs
√
x2s+y2s+z2s



(4)

<!-- page 3 -->
where ri represents the range, θi represents the azimuth
angle, and ϕi represents the elevation angle.
4) 3D Polar-Elevation to 2D Polar Image Projection:
The projection function πp maps a 3D polar-elevation point
to a 2D polar representation by discarding the elevation
component:
πp(SPEpi) .= SP ˆpi =
ri
θi

(5)
a) Polar to Image Pixel Transformation: The transfor-
mation from the polar representation to image pixel coor-
dinates applies a similarity transformation that scales the
coordinates, translates the polar frame origin to the image
frame origin (located at the top-left corner), and rotates the
frame to align the image x-axis with azimuth and the y-axis
with range. This is defined as:
IPpi = SIP SP SP ˆpi =

IPri
IPθi

=


sr cos(ω)
−sθ sin(ω)
tx
sr sin(ω)
sθ cos(ω)
ty
0
0
1




ri
θi
1


(6)
where sr and sθ are scale factors, ω is a angle offset, and
tx, ty are translation offsets.
5) 3D Polar-Elevation to 2D Elevation-Azimuth Image
Projection: The projection function πe maps a 3D polar-
elevation point to a 2D elevation-azimuth representation by
discarding the range component:
πe(SPEpi) .= SE ˆpi =
ϕi
θi

(7)
a) Elevation-Azimuth to Image Pixel Transformation:
The transformation from the elevation-azimuth representa-
tion to image pixel coordinates applies a similarity transfor-
mation similar to the polar case. This is defined as:
IEpi = SIE SE SE ˆpi =

IEϕi
IEθi

(8)
=


sϕ cos(ω)
−sθ sin(ω)
tx
sϕ sin(ω)
sθ cos(ω)
ty
0
0
1




ϕi
θi
1


(9)
where sϕ and sθ are scale factors, ω is a rotation angle, and
tx, ty are translation parameters.
B. Sonar Gaussian Splatting
1) Gaussian Representation: We revised the Gaussian
representation in [4]. Each Gaussian is defined under world
frame (W) as follows:
Gi = {µi, Σi, Ii, Λi}
(10)
where µi
.= [x, y, z]T ∈R3 is the mean position, Σi ∈R3×3
is the covariance matrix, Ii ∈R3 is the color intensity, and
Λi ∈R is the opacity of the i-th Gaussian.
2) Mean Projection: The mean position of each Gaussian
is projected into the sonar image space following the Sonar
Geometry in II-A.
Elevation 
Direction
Range 
Direction
Transmittance 
Pre-Calculation  
Polar Image 
Plane 
Elevation-Azimuth 
Image Plane 
Intensity 
Accumulation  
Fig. 3: Two-Ways Sonar Splat Visualization.
a) Polar Image: For polar images, the projected mean
in pixel coordinates is obtained by the complete transforma-
tion chain:
IPµi = SIPSP πp(πpe(TSW µi))
(11)
b) Elevation-Azimuth Image:
For elevation-azimuth
images, the projected mean is obtained by:
IEµi = SIESE πe(πpe(TSW µi))
(12)
3) Covariance Projection: To propagate uncertainty from
the world frame through the projection pipeline, we compute
the projected covariance in image space for both polar
and elevation-azimuth representations. The transformation
matrices used in covariance projection exclude the translation
components, denoted by ˆSIPSP and ˆSIESE, as translations do
not affect the spread of uncertainty.
a) Polar Image: The projected covariance for polar
images is computed as:
IPΣi = ˆSIPSPJπpJπpeRSWΣiRT
SWJT
πpeJT
πp ˆST
IPSP
(13)
where ˆSIPSP is the 2 × 2 upper-left submatrix of SIPSP that
excludes the translation component, Jπpe is the Jacobian
of the Cartesian to polar-elevation conversion function πpe
(mapping from Spi to SPEpi), Jπp is the Jacobian of the
polar projection function πp (mapping from SPEpi to SP ˆpi),
RSW is the rotation from world to sonar frame, and Σ is the
covariance in the world frame.
b) Elevation-Azimuth Image: The projected covariance
for elevation-azimuth images is computed as:
IEΣi = ˆSIESEJπeJπpeRSWΣiRT
SWJT
πpeJT
πe ˆST
IESE
(14)
where ˆSIESE similarly represents the 2 × 2 upper-left subma-
trix of SIESE without the translation terms, Jπpe is the Jaco-
bian of the Cartesian to polar-elevation conversion function
πpe (mapping from Spi to SPEpi), and Jπe is the Jacobian of
the elevation-azimuth projection function πe (mapping from
SPEpi to SE ˆpi).
4) Two-Ways Splatting: The critical differences between
the proposed sonar Gaussian splatting and visual Gaussian
splatting [4] lie in the splatting and rendering process, which
must account for the unique characteristics of sonar imaging.
The original camera-based model performs splatting and
alpha blending both along the depth (Z) direction, making

<!-- page 4 -->
transmittance calculation straightforward. However, in sonar
imaging, the intensity values are accumulated along the ele-
vation direction (corresponding to the polar image frame IP)
while the transmittance calculation must be performed along
the range direction (corresponding to the elevation-azimuth
image frame IE). This dual-direction requirement introduces
additional computational complexity during rendering.
To address this challenge, we propose Two-Ways Splat-
ting, which incorporates the intensity accumulation and
transmittance calculation simultaneously. The rendering pro-
cess can be described as:
I =
X
i∈N
Iiαi
|
{z
}
polar image frame IP
Y
j∈M
(1 −ˆαj)
|
{z
}
elevation-azimuth image frame IE
(15)
where αi is the opacity of the i-th Gaussian evaluated on
the 2D polar image frame IP, and N is the set of Gaussians
projected onto the current pixel along the elevation arc. The
term Q
j∈M(1 −ˆαj) represents the transmittance, which
is pre-computed on the elevation-azimuth image frame IE,
where M denotes the set of Gaussians located before the
current Gaussian along the range direction, and ˆαj is the
opacity of the j-th Gaussian evaluated on the elevation-
azimuth image frame.
In the proposed Two-Ways Splatting, we divide the splat-
ting process into two steps:
First, we splat the Gaussians onto the frame IE to pre-
compute the transmittance along the range direction at each
Gaussian’s 2D mean position IEµi. This procedure com-
putes only one transmittance value per Gaussian, which
is subsequently used during alpha blending. Although this
approach approximates the transmittance for all pixels within
a Gaussian’s footprint, it significantly reduces computational
cost compared to repeatedly calculating transmittance for
each pixel during polar image splatting.
Second, after obtaining the transmittance, we perform
alpha blending in the frame IP. This process follows the
standard Gaussian Splatting pipeline to accumulate intensity
along the elevation direction while incorporating the pre-
computed transmittance values as in (15).
C. Sonar Noise Modeling with Gaussian Mixture Models
Sonar imaging is inherently affected by complex noise
patterns that arise from the interaction between acoustic
beams and the underwater environment. To model this noise,
we employ a Gaussian Mixture Model (GMM) that captures
the characteristics of sonar beam patterns in both azimuth
and range dimensions as shown in Figure 4. Our approach
learns image-specific noise distributions that can be applied
during rendering to synthesize realistic sonar imagery.
1) GMM Formulation: We model the noise distribution
for each sonar image using two independent GMMs: one
for the azimuth (cross-range) direction and one for the
range (along-track) direction. For an image with index n ∈
{1, . . . , N}, the complete noise model is defined by:
Azimuth
Range
Range
Azimuth
Intensity
Intensity
Range
Azimuth
Noise Intensity
Range GMM
GMM Noise Modeling
Combine
Azimuth GMM
Multi-Path 
Reflection
Speckle 
Noise
Sidelobe 
Noise
Fig. 4: Proposed GMM Noise Model for Sonar Imaging.
a) Azimuth GMM: The azimuth noise distribution is
modeled per range bin using K Gaussian components:
p(θ|h, n) =
K
X
k=1
π(n)
k,hN(θ; µ(n)
k,h, (σ(n)
k,h)2)
(16)
where:
• θ ∈[−FOV/2, FOV/2] is the azimuth angle
• h ∈{1, . . . , H} is the range bin index (H in total)
• µ(n)
k,h ∈R and σ(n)
k,h ∈R+ are the mean and the standard
deviation for component k at row h
• π(n)
k,h
∈
[0, 1]
is
the
mixing
coefficient
with
PK
k=1 π(n)
k,h = 1
Additionally, a gain parameter g(n)
h
∈R+ per range bin
modulates the overall noise intensity.
b) Range GMM: Similarly, the range noise distribution
is modeled per azimuth bin using K Gaussian components:
p(r|w, n) =
K
X
k=1
˜π(n)
k,wN(r; ˜µ(n)
k,w, (˜σ(n)
k,w)2)
(17)
where:
• r ∈[0, Rmax] is the range distance
• w ∈{1, . . . , W} is the azimuth bin index (W in total)
• ˜µ(n)
k,w ∈R and ˜σ(n)
k,w ∈R+ are the mean range and the
standard deviation for component k at column w
• ˜π(n)
k,w
∈
[0, 1]
is
the
mixing
coefficient
with
PK
k=1 ˜π(n)
k,w = 1
Similarly, it is associated with a gain parameter ˜g(n)
w
learned
per azimuth bin.
2) Noise Generation: When rendering a pixel (n, h, w) at
a particular position on an image, we sample noise values
from both azimuth and range GMMs to generate the final
noise contribution.

<!-- page 5 -->
a) Azimuth Noise:
Nθ(h, w) = g(n)
h
K
X
k=1
π(n)
k,h exp
 
−
(θw −µ(n)
k,h)2
2(σ(n)
k,h)2
!
(18)
b) Range Noise:
Nr(h, w) = ˜g(n)
w
K
X
k=1
˜π(n)
k,w exp
 
−
(rh −˜µ(n)
k,w)2
2(˜σ(n)
k,w)2
!
(19)
c) Combined Rendering:
The final rendered image
combines the clean splatted output Iclean with both noise
components:
Ifinal(h, w) = Iclean(h, w) + Nθ(h, w) + Nr(h, w)
(20)
D. Pipeline Overview
Figure 1 illustrates the overall pipeline of the proposed
NAS-GS method. Our training can be divided into two
stages. The first stage focuses on initially optimizing the 3D
GS for the scene geometry. The second stage incrementally
incorporates the proposed GMM-based noise model to refine
the Gaussians, while learning the GMM noise distributions.
In the first stage, we first initialize GS from polar images
where pixel intensity is greater than a threshold. Then we
project these GS into the sonar image space using the
Sonar Geometry transformations. The rendering is performed
with the proposed Two-Ways Splatting technique that first
calculates transmittance along range directions for each
GS efficiently before accumulating pixel intensities along
elevation direction. Different from the original 3D GS [4]
which uses single-view rendering loss to optimize the GS,
we use a multi-view loss. This is because the scene geometry
is consistently represented across views in sonar imaging,
while complex noise randomly appears depending on the
viewpoint. Therefore, using multi-view loss can help the
Gaussians focus on fitting the underlying geometry rather
than noises in individual views.
In the second stage, the GMM-based noise model is intro-
duced to capture sonar noises. Each model is parameterized
by learnable means, variances, and mixing coefficients for
GMM components. During training, we jointly optimize
the 3D GS parameters and the GMM parameters using a
combined loss function that includes a reconstruction loss
between rendered and reference images. This allows the GS
to fit the true scene geometry while the GMM captures the
complex noise characteristics inherent in sonar imaging.
III. EXPERIMENTAL RESULTS
A. Environment Setup
1) Datasets: Both simulated and real-world offshore un-
derwater sonar datasets are used for evaluation.
a) Simulation Dataset: The simulation dataset is gen-
erated using the ray-based multi-beam sonar simulator [13]
since it can produce high-fidelity sonar images. Three dif-
ferent scenes (Cabinet, Barrel, Panel) are created in the
simulator, each of which is scanned using a Blueview p900
sonar model with 90 degree horizontal field of view and
20 degree vertical field of view. The sonar is equipped on
a robot platform traveling around four sides of the objects
and captures over 100 sonar images with a resolution of
512x399 pixels. The sonar pose data is also obtained from
the simulator.
b) Offshore Dataset: The offshore dataset was collected
using an underwater robot equipped with a Tritech Gemini
1200ik imaging sonar at an offshore wind turbine foundation
with a height of approximately 50 meters. The robot is also
equipped with a stereo camera, an IMU, and a DVL. A
long sequence with a over 20 minutes return trajectory from
the surface to the seabed is used to cover various types of
real noise (side-lobe, speckle, and multi-path noise) present
in the sonar images for evaluation in challenging offshore
large-scale scenarios. The same set of 6-DoF poses estimated
through Aqua-SLAM [14] is used for training.
2) Competing Methods and Evaluation Metrics: We com-
pare our method against two baseline approaches. Neusis [6]
is a neural implicit surface reconstruction method designed
for sonar data that uses signed distance functions and volume
rendering. ZSplat⋆[11] is an adaptation of the camera-
sonar 3D Gaussian Splatting method whose camera model
is disabled for sonar-only evaluation.
3) Gaussian Initialization: For the simulation dataset, we
use simulated lidar scans to generate initial Gaussians for
NAS-GS and ZSplat⋆training since ZSplat requires initial-
ization from point clouds. However, our method supports
initialization directly from sonar images. Therefore, in the
offshore dataset, we directly initialize Gaussians using sonar
images with pixel intensity above a certain threshold to
demonstrate the practicality of our approach in real-world
scenarios where prior point cloud may not be available.
4) Evaluation Metrics: For novel view synthesis, render-
ing quality is evaluated using the standard metrics introduced
in [4]: Peak Signal-to-Noise Ratio (PSNR), Structural Sim-
ilarity Index (SSIM) and Learned Perceptual Image Patch
Similarity (LPIPS). For 3D reconstruction, Chamfer Distance
(CD) and Hausdorff Distance (HD) [6] are used to assess 3D
geometric accuracy.
5) Hardware: All experiments are conducted on a desktop
PC equipped with a NVIDIA RTX 4070 GPU (12GB), AMD
Ryzen 7 7800X3D CPU and 32GB RAM.
B. Evaluation using Simulation Data
1) Sonar Image Rendering and View Synthesis: The quan-
titative results of the sonar novel view synthesis performance
are summarized in Table I, and the qualitative comparisons
are shown in Figure 5. Our method achieves superior per-
formance across all metrics compared to the baselines. We
consistently obtain PSNR values above 37 dB, SSIM scores
near 0.98, and LPIPS values below 0.05, demonstrating both

<!-- page 6 -->
GT
Cabinet
Barrel
Panel
Ours
Neusis
ZSplat*
Ours
(Denoise)
Fig. 5: Qualitative Comparison of Rendered Sonar Images on the Simulation Dataset.
TABLE I: Evaluation on Novel View Synthesis.
Ours
Neusis
ZSplat⋆
Scene
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Cabinet 38.95
0.99
0.03
32.78
0.92
0.09
29.73
0.97
0.05
Barrel
37.95
0.98
0.04
30.87
0.89
0.10
28.73
0.97
0.05
Panel
37.42
0.98
0.04
29.79
0.85
0.17
27.04
0.96
0.08
Note: Best results are bolded, second-best results are underlined.
high pixel-wise accuracy and perceptual quality. The ”Ours
(Denoise)” variant, which removes the learned GMM noise
during rendering, shows that our GMM-based noise model
effectively captures the sonar-specific artifacts and helps
the 3D Gaussian representation focus on the real geometry.
Neusis does not capture fine details due to its lack of noise
handling, resulting in PSNR values around 30-33 dB and
noticeably lower SSIM scores. ZSplat⋆, while achieving
better structural similarity than Neusis, shows sparse and
blurred rendering results with PSNR values around 27-30 dB
compared to our method. The qualitative results in Figure
5 clearly demonstrate the effectiveness of the proposed
approach. It preserves sharp edges and fine structures while
accurately reproducing the characteristic sonar noise patterns.
The denoised version reveals the underlying geometric struc-
ture rendered by the 3D GS beneath the learned GMM noise
model.
2) Sonar based 3D Reconstruction: To convert the 3D
Gaussian representation to a mesh, we first extract a dense
point cloud by sampling points from each Gaussian distribu-
tion within three standard deviations of its mean, retaining
only points where the evaluated density exceeds a predefined
threshold. We then apply the marching cubes algorithm [15]
to reconstruct a surface mesh from this point cloud.
TABLE II: Evaluation on 3D Reconstruction
Ours
Neusis
ZSplat⋆
Scene
CD↓
HD↓
CD↓
HD↓
CD↓
HD↓
Cabinet
0.006
0.054
0.277
1.333
0.298
0.871
Barrel
0.007
0.012
2.096
6.676
0.020
0.320
Panel
0.060
0.464
0.313
2.633
0.342
1.249
Note: CD = Chamfer Distance, HD = Hausdorff Distance. Best
results are bolded, second-best results are underlined.
The quantitative results comparing the reconstructed
meshes against ground truth models are presented in Table
II. Our method demonstrates accurate geometric reconstruc-
tion across all scenes, achieving Chamfer Distances on the
order of 0.005-0.060 and Hausdorff Distances below 0.5,
representing an order of magnitude improvement over the
baseline methods in most cases. Neusis and ZSplat⋆produce
reconstructions with substantial geometric errors.
The qualitative mesh comparisons in Figure 6 further
illustrate these differences. Our reconstructed meshes exhibit
surface structures with reasonable geometric details that
match the ground truth models. In contrast, Neusis meshes
show structures with wave-shaped distortions on the edges,
overfitting to the noise, while ZSplat⋆meshes contain visible
holes and irregular surfaces due to the lack of handling sonar-
specific imaging characteristics.
C. Evaluation using Real Offshore Data
We further validate the proposed method using real-world
offshore data for sonar based novel-view synthesis and large-
scale reconstruction. The sequence spans over 20 minutes
and covers a large-scale trajectory around an foundation of
an wind turbine located North Sea. This challenging scenario

<!-- page 7 -->
GT
Ours
Neusis
ZSplat⋆
Cabinet
Barrel
Panel
Fig. 6: Qualitative Comparison of Sonar 3D reconstruction.
Fig. 7: Offshore novel view synthesis results.
features various types of complex noise patterns, including
side-lobe artifacts, speckle noise, and multi-path reflections
that are characteristic of real offshore environments.
We compared our sonar 3D reconstruction results against
the vision-based reconstruction using Block Matching(BM)
stereo matching. Notably, we only initialize GS from sonar
images where intensity is greater than a threshold, without
using any prior point cloud. And poses from Aqua-SLAM
[14] are used for both methods.
Figure 7 shows representative frames from three distinct
regions: a ladder structure on the foundation, an anode, and
a towing hole structure. The proposed method can render
realistic sonar images that visually match the ground truth
observations. Meanwhile, the denoised version reveals the
clean geometric structure, demonstrating that our GMM-
based noise model effectively separates the true geometry
from sonar-specific artifacts.
Figure 8 presents the sonar-only reconstructed 3D model
of the structure, compared with vision-based reconstruction
[14]. Despite the challenging conditions and large scale of
the scene, our reconstruction captures the complex geometric
Fig. 8: Sonar vs Camera Reconstruction on Offshore Dataset.
Camera images are not used for sonar reconstruction.
features of the underwater structure, including the ladder,
anodes, and various structural details. Notably, the recon-
struction of the algae-covered ladder demonstrates superior
quality compared to vision-based results. This is because
the vision-based method is compromised by the dynamic
movement of the algae, whereas the acoustic noise from
the algae is minimal and effectively modeled by our GMM,
resulting in a cleaner reconstruction from sonar data.
D. Computation
We evaluate the computational efficiency of our method
by measuring the rendering speed in FPS on the simulation
dataset. The rendering speed is highly dependent on the
number of 3D Gaussians used to represent the scene. In our
experiments, we load the trained models with approximately
10000 Gaussians which already provide high-quality render-
ing and reconstruction results for our method. ZSplat loads
the same ply file to ensure the same number of Gaussians
are used. Table III presents the rendering performance com-
parison across all three methods.
Our method achieves real-time rendering (>700 FPS),
approx. 2× faster than ZSplat⋆and 8,000× faster than
Neusis. Ours⋆(pure GS without GMM) reaches ∼1000 FPS.
This efficiency stems from our Two-Ways Splatting, which
pre-computes transmittance in the elevation-azimuth frame
before fast alpha blending in the polar frame. In contrast,
ZSplat⋆computes transmittance repeatedly for each pixel
during rendering, while Neusis requires computationally
expensive ray marching through implicit surface representa-
tions. The superior rendering efficiency of our method makes
it particularly suitable for large-scale offshore environments.
E. Ablation Study
To further analyze the contributions of our proposed
components, we conduct an ablation study on the simulation
dataset. We evaluate the impact of the GMM-based noise
modeling on both novel view synthesis and 3D reconstruction
performance.
1) Novel View Synthesis Ablation: As shown in Figure
9, we compare the full NAS-GS method with a variant
that excludes the GMM-based noise modeling. The results

<!-- page 8 -->
TABLE III: Rendering Performance Comparison
Scene
Ours
Ours⋆
Neusis
ZSplat⋆
FPS↑
FPS↑
FPS↑
FPS↑
Cabinet
735.98
962.01
0.09
411.62
Barrel
832.36
1032.04
0.08
419.45
Panel
803.12
1274.23
0.08
417.66
Note: Ours stands for full model, Ours⋆stands for w/o GMM.
Best results are bolded, second-best results are underlined.
Fig. 9: Ablation study on GMM-based noise modeling for
novel view synthesis.
clearly demonstrate that the GMM noise model significantly
enhances rendering quality. The rendered image with GMM
closely matches the ground truth, and the denoised ren-
dering (removing GMM noise to isolate pure GS output)
confirms that the GMM effectively captures sonar-specific
noise patterns while allowing 3D Gaussians to focus on
representing the underlying scene structure. Without the
GMM noise model, rendering quality degrades substantially.
Since sonar noise varies randomly across viewpoints, the 3D
Gaussians alone cannot consistently model noise across all
views, resulting in blurred and distorted renderings.
2) 3D Reconstruction Ablation: As shown in Figure 10,
we evaluate the impact of GMM-based noise modeling on
3D reconstruction accuracy. Without the GMM noise model,
reconstruction quality deteriorates, exhibiting distorted sur-
faces as the 3D Gaussians overfit to noise. This highlights
the importance of accurately modeling sonar noise, enabling
3D Gaussians to focus on capturing true geometric features
rather than overfitting to noise artifacts. The results indicate
that GMM noise modeling contributes significantly to im-
proved reconstruction performance.
IV. CONCLUSIONS
This paper introduces NAS-GS, a noise-aware sonar Gaus-
sian splatting framework that addresses the unique chal-
lenges of sonar reconstruction and novel view synthesis. The
proposed two-ways Splatting technique efficiently handles
sonar’s polar imaging geometry, while the GMM-based noise
modeling effectively captures complex sonar-specific noise
Fig. 10: Ablation study on GMM-based noise modeling on
3D reconstruction.
patterns and allow the 3D Gaussians to focus on representing
the underlying scene structure. Extensive experiments on
both simulated and real-world offshore datasets demonstrate
that NAS-GS outperforms state-of-the-art methods in novel
view synthesis and 3D reconstruction tasks, achieving high-
fidelity results with real-time rendering speeds. Future work
will explore extending the framework to sonar SLAM and
dynamic scenes and integrating temporal coherence for im-
proved reconstruction of time-varying underwater environ-
ments.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp.
99–106, 2021.
[2] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in
Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5470–5479.
[3] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5501–5510.
[4] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[5] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5354–5363.
[6] M. Qadri, M. Kaess, and I. Gkioulekas, “Neural implicit surface
reconstruction using imaging sonar,” in 2023 IEEE International
Conference on Robotics and Automation (ICRA).
IEEE, 2023, pp.
1040–1047.
[7] Y. Xie, G. Troni, N. Bore, and J. Folkesson, “Bathymetric surveying
with imaging sonar using neural volume rendering,” IEEE Robotics
and Automation Letters, vol. 9, no. 9, pp. 8146–8153, 2024.
[8] T. Lin, M. Qadri, K. Zhang, A. Pediredla, C. A. Metzler, and M. Kaess,
“Acoustic neural 3d reconstruction under pose drift,” arXiv preprint
arXiv:2503.08930, 2025.
[9] Y. Feng, W. Lu, H. Gao, B. Nie, K. Lin, and L. Hu, “Differentiable
space carving for 3d reconstruction using imaging sonar,” IEEE
Robotics and Automation Letters, vol. 9, no. 11, pp. 10 065–10 072,
2024.
[10] A. V. Sethuraman, M. Rucker, O. Bagoren, P.-C. Kung, N. N. Amutha,
and K. A. Skinner, “Sonarsplat: Novel view synthesis of imaging sonar
via gaussian splatting,” arXiv preprint arXiv:2504.00159, 2025.
[11] Z. Qu, O. Vengurlekar, M. Qadri, K. Zhang, M. Kaess, C. Metzler,
S. Jayasuriya, and A. Pediredla, “Z-splat: Z-axis gaussian splatting
for camera-sonar fusion,” IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2024.
[12] Z. Ling, Y. Feng, A. Meng, R. Xiao, S. Pan, W. Lu, and L. Hu,
“Aqua-splat: Physically-informed sonar-camera gaussian splatting for
underwater 3d reconstruction,” IEEE Robotics and Automation Letters,
2025.

<!-- page 9 -->
[13] W.-S. Choi, D. R. Olson, D. Davis, M. Zhang, A. Racson, B. Bingham,
M. McCarrin, C. Vogt, and J. Herman, “Physics-based modelling
and simulation of multibeam echosounder perception for autonomous
underwater manipulation,” Frontiers in Robotics and AI, vol. 8, p.
706646, 2021.
[14] S. Xu, K. Zhang, and S. Wang, “Aqua-slam: Tightly-coupled un-
derwater acoustic-visual-inertial slam with sensor calibration,” IEEE
Transactions on Robotics, 2025.
[15] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution
3d surface construction algorithm,” in Seminal graphics: pioneering
efforts that shaped the field, 1998, pp. 347–353.
