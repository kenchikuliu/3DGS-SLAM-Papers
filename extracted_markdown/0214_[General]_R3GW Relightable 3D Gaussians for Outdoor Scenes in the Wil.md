<!-- page 1 -->
R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild
Margherita Lea Corona1 , Wieland Morgenstern1 , Peter Eisert1,2
and Anna Hilsmann1
1Fraunhofer Heinrich Hertz Institute, Berlin, Germany
2Humboldt Universit¨at zu Berlin, Berlin, Germany
{margherita.lea.corona, wieland.morgenstern, peter.eisert, anna.hilsmann}@hhi.fraunhofer.de
Keywords:
Gaussian Splatting, 3D Reconstruction, Physically Based Rendering, Relighting.
Abstract:
3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel
view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method
does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS
struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing
lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS rep-
resentation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable
foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW mod-
els view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering
with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively
and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for
physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under
arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction
artifacts, improving rendering quality at the sky-foreground boundary.
Figure 1: R3GW reconstructs a relightable 3D Gaussian
Splatting representation of an outdoor scene from a col-
lection of unconstrained multiview images.
Our method
factorizes the scene’s foreground into geometry, material,
and illumination, with the lighting represented as an envi-
ronment map. This enables illumination editing along with
novel view synthesis. The sky, as a non-reflective surface,
is modeled with a dedicated set of Gaussians that remain
independent of scene lighting and material.
1
INTRODUCTION
In recent years, 3D Gaussian Splatting (3DGS) (Kerbl
et al., 2023) has emerged as a revolutionary tech-
nique for 3D reconstruction and novel view synthe-
sis of static scenes captured by a collection of mul-
tiview images.
Unlike prior approaches based on
NeRF (Mildenhall et al., 2020), which represent the
scene volume implicitly with a multi-layer percep-
tron (MLP), 3DGS uses a discrete set of 3D Gaus-
sian primitives. Each Gaussian is parameterized by
learnable attributes that encode the geometry and ap-
pearance. To render the scene, 3DGS utilizes an ef-
ficient and differentiable rasterization algorithm. In
contrast to NeRF, the explicit representation and ren-
dering strategy of 3DGS allow for fast training and
real-time rendering while maintaining high visual fi-
delity. Despite its success, the method presents a ma-
jor limitation. 3DGS does not disentangle the lighting
from the material and geometry of the scene, prevent-
ing relighting under novel illumination conditions and
thus restricting the approach to the task of novel view
synthesis.
Given the growing demand for control-
lable, photorealistic scene representations, e.g. for im-
mersive AR/VR experiences, extending 3DGS to en-
able relighting has recently become a key challenge
in the research community.
Recent works (Gao et al., 2024; Liang et al., 2024;
Jiang et al., 2024) introduced a relightable 3DGS
scene representation by integrating Physically Based
arXiv:2603.02801v1  [cs.CV]  3 Mar 2026

<!-- page 2 -->
Rendering (PBR) into the rendering pipeline of the
method, enabling the decomposition of the scene into
lighting, material, and geometry. However, these ap-
proaches are restricted to the representation of static
environments with fixed illumination across the im-
ages. As a result, they do not generalize to scenes cap-
tured in the wild by unconstrained photo collections,
which are characterized by transient objects and, par-
ticularly, by changing illumination.
In this paper, we present R3GW, a method that
extends 3DGS to support relighting of outdoor scenes
captured in the wild. An outdoor environment pho-
tographed in an unconstrained setting, in addition to
its varying illumination, poses a unique challenge.
The sky, unlike the foreground objects, is not a re-
flective surface, and its appearance is not governed
by light-material interaction. Our approach addresses
this with a decoupled representation, separately mod-
eling the appearance of the foreground and the sky
with two distinct sets of Gaussians with different at-
tributes. For the foreground, R3GW learns the vary-
ing illumination of the scene per image, represented
by an environment map and parameterized by Spher-
ical Harmonics (SH). For reflection, a Cook-Torrance
BRDF parameterized following the Disney material
model (Cook and Torrance, 1982; Burley, 2012) is
used. The sky appearance, on the other hand, is mod-
eled independently of the environment light by pa-
rameterizing the sky color with low-degree SH, which
are learned separately per image.
Our contributions are summarized as follows:
1. We present a new 3D Gaussian Splatting frame-
work that handles outdoor scenes captured in the
wild and supports relighting under arbitrary envi-
ronment maps.
2. We integrate PBR with a Cook-Torrance BRDF
into the 3DGS scene representation in a varying
illumination setting. To achieve this, we lever-
age the split-sum approximation technique (Karis,
2013) and the SH representation of the environ-
mental light.
3. We propose a decoupled sky-foreground represen-
tation for the 3DGS framework. The sky is rep-
resented as a set of Gaussians constrained to the
background region of the upper hemisphere en-
closing the scene. This separation yields a phys-
ically plausible scene representation, where the
sky appearance remains independent of surface
material. Moreover, the design of the sky region
enhances the scene depth estimation and conse-
quently improves the rendering of elements lo-
cated at the sky-foreground boundary.
2
RELATED WORK
2.1
Relightable 3D Gaussian Splatting
for Static Scenes
Recently, (Gao et al., 2024; Liang et al., 2024; Chen
et al., 2024; Wu et al., 2025) have adapted 3DGS
to support both novel view synthesis and relighting.
These methods enable relighting by explicitly model-
ing reflections using PBR, either by integrating it di-
rectly into the color attribute of the Gaussians (Gao
et al., 2024) or by applying deferred shading after
rasterization (Liang et al., 2024; Chen et al., 2024;
Wu et al., 2025). The incoming illumination of the
scene is modeled as a global learnable environment
map and the 3DGS attributes are augmented with sur-
face normals and BRDF material features. Gaussian
Shader (Jiang et al., 2024) proposes a similar ap-
proach by adopting a simplified shading model, and,
unlike (Gao et al., 2024; Liang et al., 2024; Chen
et al., 2024; Wu et al., 2025), it does not model in-
direct lighting to account for soft shadows and in-
terreflection effects.
Despite enabling high-quality
novel view synthesis under arbitrary lighting condi-
tions, these methods rely on the assumption of static
lighting. This makes them unsuitable for editing the
illumination of scenes captured in an unconstrained
environment. Our approach addresses this limitation
by repurposing PBR combined with 3DGS in an in-
the-wild setting.
2.2
3D Gaussian Splatting for Outdoor
Scenes in the Wild
A number of methods have tailored 3DGS for un-
constrained outdoor environments, addressing the ap-
pearance variations using per-image and per-Gaussian
embeddings (Kulhanek et al., 2024; Zhang et al.,
2024; Xu et al., 2024).
Nevertheless, these ap-
proaches focus on removing transient objects and
do not enable relighting.
Similar to our work,
WildGaussians (Kulhanek et al., 2024) augments the
initial set of Gaussians with sky Gaussians sampled
on the sphere around the scene.
However, unlike
our method, these Gaussians are neither tracked nor
guided to effectively represent the sky.
2.3
Relightable 3D Representation of
Outdoor Scenes in the Wild
Several approaches have extended the neural scene
representation (NeRF) of (Mildenhall et al., 2020) to
enable relighting of outdoor scenes in the wild. These

<!-- page 3 -->
methods typically assume a diffuse reflectance model,
where the color of a point of the scene surface is in-
dependent of the viewing direction and determined by
the interaction of incident light, albedo, surface nor-
mal, and shadows.
NeRF-OSR (Rudnev et al., 2022) learns the inci-
dent illumination per image, represented as an envi-
ronment light parameterized by low-degree SH, and
models albedo, surface normals, and shadows with
additional MLPs. SOL-NeRF (Sun et al., 2023) uti-
lizes a different lighting representation, modeling the
scene illumination as a combination of a skylight, pa-
rameterized by SH, and sunlight, parameterized by
Spherical Gaussians.
The method additionally in-
corporates a prior on the sunlight color. Similarly,
NeuSky (Gardner et al., 2024) regularizes the learned
environment light of the scene with a neural illumi-
nation prior. Compared to earlier deep-learning ap-
proaches for outdoor scene relighting (Philip et al.,
2019; Yu et al., 2020), these methods improve ren-
dering quality and enable editing both the camera
viewpoint and illumination. However, due to the im-
plicit scene representation of NeRF, they suffer from
slow training and rendering.
Moreover, these ap-
proaches only model diffuse reflections, neglecting
view-dependent lighting effects.
In contrast, FEGR (Wang et al., 2023) uses a hy-
brid deferred renderer that combines the volumetric
rendering of the neural field with mesh extraction
to reproduce view-dependent specular highlights and
shadows. Nonetheless, the approach remains costly to
train. On the other hand, SR-TensoRF (Chang et al.,
2024), building upon (Chen et al., 2022a), signifi-
cantly reduces the training time, but still relies on the
assumption of low-frequency lighting.
LumiGauss (Kaleta et al., 2025) introduces a re-
lightable 2D Gaussian Splatting (Huang et al., 2024)
representation for outdoor scenes captured in the
wild.
The method assumes a diffuse reflectance
model and enables relighting by learning per-image
environment light parameterized by second-order SH.
The color of the Gaussians is computed as a func-
tion of the environment light, albedo, surface nor-
mal, and per-Gaussian shadows modeled with SH.
LumiGauss achieves high-quality reconstruction and
relighting results while offering faster training and
rendering than the related NeRF-based techniques.
Nonetheless, the method models diffuse reflections
only and uses a single collection of Gaussians to rep-
resent both the sky and the foreground of the scene.
Our method, while not explicitly modeling shadows,
adopts a view-dependent reflectance model for the
foreground of the scene, which captures more com-
plex lighting effects in addition to diffuse reflections.
Moreover, with our decoupled representation of the
sky and foreground, we model their appearance in-
dependently, ensuring that the Gaussians representing
the sky are not assigned with material properties and
surface normals.
Concurrently with our work, GaRe (Haiyang
et al., 2025) tailors 3DGS for outdoor scene relight-
ing in the wild by enhancing the Gaussians with shad-
ing latent codes.
These latent vectors are decoded
from per-image global illumination features that cap-
ture sunlight, skylight, and indirect lighting effects.
Unlike our method, GaRe does not enable relighting
under arbitrary environment maps due to its implicit
illumination model. Furthermore, the approach still
relies on the assumption of Lambertian surfaces.
3
PRELIMINARIES ON 3D
GAUSSIAN SPLATTING
Our method builds upon 3D Gaussian Splatting
(3DGS) (Kerbl et al., 2023). 3DGS represents a static
scene as a set of 3D Gaussian primitives with opti-
mizable attributes: a position p = (x,y,z)T ∈R3, a co-
variance matrix Σ ∈R3×3, an opacity value o ∈[0,1],
and an RGB view-dependent color c ∈[0,1]3 param-
eterized by SH. For efficient optimization, the co-
variance matrix is factorized as Σ = RSSTRT, where
R is a rotation matrix describing the orientation of
a Gaussian in world space, while the scaling ma-
trix S = diag(sx,sy,sz) encodes extensions along each
axis. Each Gaussian is then associated with a Gaus-
sian density function, defined for every u ∈R3 as:
g(u|p,Σ) ∝e−1
2 (u−p)T Σ−1(u−p).
(1)
To render the scene, 3DGS uses a fast differen-
tiable tile-based rasterizer implemented in CUDA.
The visible Gaussians are first projected (splatted)
onto the 2D image plane of the given camera view.
After projection, the color of each pixel in the output
image is computed by sorting the splats by depth and
accumulating their colors via α-blending:
C =
N
∑
i=1
ciαi
i−1
∏
j=1
(1−αj),
(2)
where N denotes the number of visible splats and the
blending weights αi are defined as the product of the
Gaussian splat opacity with the 2D Gaussian density
function evaluated at the pixel value.
The optimization of a 3D Gaussian Splatting
scene begins with a collection of RGB images paired
with calibrated cameras obtained from COLMAP
SfM (Schonberger and Frahm, 2016). The positions

<!-- page 4 -->
of the Gaussians are initialized with the sparse point
cloud produced by SfM. All the Gaussians attributes
are optimized by minimizing the reconstruction loss
Lrec = λL1 + (1 −λ)LD-SSIM, λ > 0, between the
ground truth images and their renderings. During the
first training stage, the number of Gaussians is adap-
tively controlled. The Gaussians in over and under-
reconstructed regions are cloned, while those that do
not enhance the rendering quality are pruned.
4
METHOD
R3GW represents the foreground and the sky of an
outdoor scene captured in the wild using two sepa-
rate sets of 3D Gaussians: the foreground and the sky
Gaussians. Fig. 2 overviews our method.
To enable relighting, our approach explicitly mod-
els the incoming illumination of the scene as an
RGB environment light, represented by an equirect-
angular environment map and compactly parameter-
ized by SH. Diffuse lighting is typically modeled
using second-order SH. Since our goal is to ac-
count for additional view-dependent effects, we use
a higher degree of llight = 4.
We denote the cor-
responding SH coefficients by llight ∈R3×(llight+1)2 ,
and the reconstructed environment light as L(·) ∈
[0,∞)3. R3GW learns the environment light based
on the foreground pixels of the training images. It
models the foreground reflections assuming a Cook-
Torrance microfacet-based reflectance model (Cook
and Torrance, 1982) and integrates PBR into the
3DGS representation by computing the color of the
foreground Gaussians using the rendering equation
(Kajiya, 1986). In contrast, the sky is modeled inde-
pendently of material properties and surface normals,
as its appearance is not influenced by surface interac-
tions. Therefore, we fix the color of the sky relative
to the distant lighting and parameterize it with first-
order SH, which we denote as lsky ∈R3×(lsky+1)2, with
lsky = 1. Sky Gaussians’ colors are derived from lsky.
4.1
Varying Illumination and Sky
Appearance
To handle the varying illumination and sky appear-
ance of the scene, following (Kaleta et al., 2025), each
image is associated with a learnable embedding vec-
tor e encoding its appearance information. The en-
vironment light and the sky color SH coefficients are
predicted per image by an MLP:
llight,lsky = MLP(e|Θ),
(3)
where Θ is the matrix of the MLP weights, which are
optimized alongside the images’ latent codes. During
training, the image appearance embeddings are stored
in an optimizable lookup table.
4.2
Relightable Foreground
Representation
We model the foreground Gaussians as the relightable
component of the scene. In addition to the attributes
of standard 3DGS, each Gaussian is endowed with the
surface normal at its position and with the material
properties of the Cook-Torrance BRDF parameterized
according to a simplified Disney material model (Bur-
ley, 2012): albedo a ∈[0,1]3 and roughness ρ ∈[0,1].
We assume outdoor environments to be dominated by
non-metallic surfaces and fix the metallic parameter
of the BRDF at zero.
By following (Jiang et al., 2024), the normal of a
foreground Gaussian is modeled as its shortest axis:
n = Ri∗, where Ri∗, i∗∈{1,2,3}, is the column of the
rotation matrix corresponding to the minimum scaling
coefficient.
The color of a foreground Gaussian is the light
exiting from its position along its viewing direction
ωo, which is given by the rendering equation for non-
emitting surfaces (Kajiya, 1986):
c(ωo) = a
π
Z
ΩL(ωi)(n·ωi)dωi
|
{z
}
cdiffuse
(4)
+
Z
ΩL(ωi) DFG
4(n·ωo) dωi
|
{z
}
cspecular
(5)
where Ωdenotes the upper hemisphere centered
around n and ωi is the light incident direction. The
functions D,F,G, depending on the roughness ρ, cor-
respond to the material properties given by normal
distribution of the microfacet orientations, the Fres-
nel reflection, and the geometric attenuation due to
the facets shadowing and masking, respectively. No-
tably, the color of a foreground Gaussian is the sum of
a diffuse and a specular color: cdiffuse,cspecular, where
only cspecular depends on the viewing direction ωo.
The diffuse color is computed using the approach
of LumiGauss (Kaleta et al., 2025), which follows
(Ramamoorthi and Hanrahan, 2001). By denoting as
ˆllight ∈R3×9 the SH coefficients of the environment
light up to the second degree and by A ∈R9 the SH
coefficients of the cosine term (n · ωi) in Eq. (4), we
compute the diffuse color as:
cdiffuse = a
π ⊙(nTM(ˆllight,A)n),
(6)

<!-- page 5 -->
Figure 2: Training pipeline of R3GW. R3GW learns a relightable 3DGS representation of an outdoor scene captured in the
wild. The PBR color of the foreground Gaussians, depending on the surface normals and material properties at their positions,
as well as on the environment light, enables relighting of the scene’s foreground. In contrast, the color of the sky Gaussians is
independent of the scene illumination. The rendered image is formed by rasterizing the sky and the foreground Gaussians in a
single pass. The Gaussians are regularized so that the sky Gaussians are responsible for the rendering of the sky pixels, while
the foreground Gaussians contribute only to the foreground region. After training, the foreground Gaussians can be rendered
under novel lighting conditions by supplying the corresponding environment map as input.
where M is a symmetric matrix. The definition of M,
which is derived from ˆllight and A, can be found in
(Ramamoorthi and Hanrahan, 2001). For the diffuse
color we use only the second-order SH coefficients of
the environment light since, as shown in (Ramamoor-
thi and Hanrahan, 2001), the SH coefficients of the
cosine term tend to 0 rapidly after degree 2.
Following (Jiang et al., 2024; Gao et al., 2024),
we compute the specular color using the split-sum
approximation technique introduced in (Karis, 2013),
which makes the integral in Eq. (5) tractable:
cspecular ≈
Z
ΩL(ωi)D(ρ,r)dωi
|
{z
}
Ispecular
Z
Ω
DFG
4(n·ωo) dωi
|
{z
}
IBRDF
. (7)
In the above equation, Ispecular is the integral of the
environment light L modulated by the normal distri-
bution D, which depends on the roughness ρ and the
reflection direction r = 2(ωo · n)n −ωo. The second
term, IBRDF, corresponds to the integral of the BRDF
specular component, depending on ρ and the cosine
of the angle formed by the viewing direction with the
surface normal. Given that IBRDF is independent of
the varying environment light, as proposed in (Karis,
2013), it is precomputed and stored in a 2D lookup
texture. The value of IBRDF is then obtained as:
IBRDF = 0.04F1 +F2,
(8)
where F1 and F2 are two scalars derived by interpolat-
ing the 2D lookup texture for the Gaussian’s rough-
ness ρ and cosine value.
The constant 0.04 cor-
responds to the Fresnel-Schlick reflectivity for non-
metallic surfaces. In static lighting scenarios, the in-
tegral Ispecular can be pre-integrated and stored in the
mipmap levels of a cube map (Jiang et al., 2024; Gao
et al., 2024), following the original split-sum approx-
imation method. However, under changing illumina-
tion conditions, representing the environment light for
each image as a learnable cube map, would introduce
a large number of parameters, making the approach
computationally inefficient. Instead, we adopt a sim-
plified approach by applying a smoothing operation
to the SH coefficients of the environment light llight.
Specifically, the coefficients are convolved channel-
wise with a Gaussian kernel in the SH domain. The
blurring strength is set as the square of the rough-
ness. The SH coefficients of the filter are defined as
in (Chung, 2005):
glm = gl0 = exp(−l(l +1)ρ2)
(9)
for every l ∈{0,...4},m ∈{−l,...,l}. The exponen-
tial function in the coefficients of the low-pass filter
prevents excessive blurring while effectively smooth-
ing the lighting SH. We approximate the specular
light Ispecular by evaluating the convolved SH coeffi-
cients, lspecular, in the reflection direction r. In sum-
mary, our method computes the specular color as:
cspecular = lT
specularY(r)⊙(0.04F1 +F2),
(10)
where Y(r) ∈R(llight+1)2 is the vector of the SH basis
functions evaluated in r.
Before rendering, the color of each foreground
Gaussian is gamma-corrected with γ = 2.2.
4.3
Sky Representation
Our method represents the sky as the portion of the
upper hemisphere surrounding the scene located be-
hind the foreground geometry, effectively treating it

<!-- page 6 -->
as the background of the scene. The sky Gaussians
are endowed with the attributes of vanilla 3DGS and,
to constrain them to the sky region, their position is
parameterized with spherical coordinates. The whole
set of sky Gaussians shares a learnable radius attribute
r ∈R+ and each sky Gaussian has optimizable po-
lar and azimuthal angle attributes, θ, φ, which are
clamped during training to the ranges [0, π
2] and [0,π],
respectively. The center of the hemisphere is fixed
and estimated as the midpoint of the axis-aligned
bounding box of the SfM point cloud. The color of
a sky Gaussian in its viewing direction ωo is derived
by evaluating the SH coefficients:
c(ωo) = lT
skyY(ωo).
(11)
We constrain the sky representation to a subset of
the upper hemisphere, as for most of the datasets con-
sidering the whole sphere would introduce unneces-
sary sky Gaussians that do not intersect the view frus-
tum of any training camera. However, the sky model
can be easily extended by expanding the ranges of the
sky Gaussians’ angle attributes.
To initialize the positions of the sky Gaussians, we
augment the SfM point cloud, used to initialize the
foreground Gaussians’ positions, with N additional
points sampled uniformly over the designated sky re-
gion. Specifically, we first sample the z coordinate
of each sky point, compute the corresponding polar
angle θ, and then uniformly sample φ. The number
of sky Gaussians is chosen proportionally to the esti-
mated sky distance, which is set as the initial value of
r. Similarly to (Kulhanek et al., 2024), we compute
it as the 99th percentile of the distances of the fore-
ground points from their average. See the Appendix
for a visualization of the initial set of sky Gaussians.
To track and control the sky Gaussians, each
Gaussian is assigned a fixed boolean attribute mark-
ing it as sky or foreground.
Ours
LumiGauss
Figure 3: Qualitative comparison of the depth maps pre-
dicted by our method and LumiGauss (Kaleta et al., 2025)
for the NeRF-OSR scenes. For each novel view, the ground
truth is visualized in Fig. 6.
Compared to LumiGauss,
R3GW produces more geometrically consistent depth maps
with significantly reduced artifacts in the sky and in the dis-
tant regions of the foreground.
Ours
LumiGauss
Ours
LumiGauss
Ou
LumiGauss
Figure 4: Qualitative comparison of rendering quality at the
sky-foreground boundary between our method and Lumi-
Gauss (Kaleta et al., 2025) for the NeRF-OSR scenes. For
a clear visualization, we render the novel views for both
R3GW and LumiGauss using trained illuminations. Our
sky Gaussians are rendered using the corresponding trained
sky color SH. R3GW yields sharper building outlines with
fewer halos as a result of an improved depth reconstruction
(see Fig. 3).
4.4
Losses
The learnable parameters of our scene representation
are the attributes of the sky and foreground Gaus-
sians (except for the color and the boolean identifier),
along with the appearance embeddings and MLP pa-
rameters. These are optimized by minimizing the re-
construction loss of 3DGS introduced in Sec. 3, Lrec,
combined with the following regularizations.
To constrain the environment light to take posi-
tive values, we use the regularization introduced by
(Kaleta et al., 2025):
Llight = 1
M
M
∑
i=1
∥min(0,L(ωi))∥2
2,
(12)
where {ωi}M
i=1 are incident light directions sampled
uniformly on the unit upper hemisphere.
To align the normals of the foreground Gaussians
with the surface normals, we use the normal consis-
tency loss of (Huang et al., 2024):
Lnormal =
N
∑
i=1
αi(1−nT
i N),
(13)
where {αi}N
i=1 are the blending weights of Eq. (2), ni
is the normal of the visible foreground Gaussian i, and
N is the reference surface normal estimated from the
gradient of the depth map. Furthermore, to encourage
flattening of the foreground Gaussians, we apply the
following loss from (Chen et al., 2023) to the smallest
scaling coefficient:
Lscale = ∥min(sx,sy,sz)∥1.
(14)
The sky Gaussians must be guided to contribute
only to the rendering of the sky, while the foreground
Gaussians should not be moved in the sky region.
Otherwise, the model may learn the scene illumina-
tion based on physically implausible BRDF features

<!-- page 7 -->
and sky normals. Therefore, we apply the following
loss to the sky and foreground pixels of the render-
ings ˆIfg and ˆIsky, which are obtained by rendering all
Gaussians with the foreground Gaussians’ colors set
to zero in ˆIsky, and the colors of the sky Gaussians set
to zero in ˆIfg:
Lfg-sky = L1((1−Msky)ˆIfg,0)+L1(Msky ˆIsky,0),
(15)
where Msky is a binary sky segmentation mask corre-
sponding to the view.
We allow the background to move only farther
away from the foreground. To enforce this, we intro-
duce the following loss to penalize the case in which
¯dforeground, the average depth of the visible foreground
Gaussians, exceeds the average depth of the visible
sky Gaussians ¯dsky:
Lsky-depth = e−γ( ¯dsky−¯dforeground), γ > 0.
(16)
In summary, the training objective function is:
L = Lrec +λlightLlight +λnormalLnormal
+λscaleLscale +λfg-skyLfg-sky +λsky-depthLsky-depth,
(17)
where we set: λlight = 1.0, λnormal = 0.05, λscale = 1.0,
λfg-sky = 0.5, λsky-depth = 0.005.
5
EXPERIMENTS
5.1
Dataset, Evaluation, and Baselines
To evaluate our approach, we strictly follow the eval-
uation protocol of NeRF-OSR (Rudnev et al., 2022)
and LumiGauss (Kaleta et al., 2025).
We use the
NeRF-OSR (Rudnev et al., 2022) dataset, which con-
tains eight outdoor scenes captured in the wild and
provides ground truth LDR environment maps for
each lighting condition in the test set of a scene. We
use the official data split for Ludwigskirche (lk2),
Staatstheater (st), and Landwehrplatz (lwp). For each
scene, we compute the average PSNR, MSE, MAE,
and SSIM over five testing viewpoints. We select one
view for each test lighting condition and render it us-
ing the corresponding ground truth environment map
as illumination. Due to the implementation design for
the image appearance embeddings, the MLP that pre-
dicts the SH coefficients of the sky color along with
those of the environment light is unavailable for the
test images. Therefore, when rendering the test views
with the ground truth environment maps, we fix the
color of the sky Gaussians to white. For the compu-
tation of the evaluation metrics, we mask out the oc-
cluders, sky, and vegetation using the segmentation
Albedo
Surface Normals
Ours
LumiGauss
NeRF-OSR
Figure 5: Qualitative comparison of the albedo map and sur-
face normals predicted by our method, NeRF-OSR (Rudnev
et al., 2022), and LumiGauss (Kaleta et al., 2025) for the
NeRF-OSR scene st. The ground truth novel view is visu-
alized in Fig. 6. R3GW synthesizes sharper surface normal
maps than LumiGauss and NeRF-OSR. Furthermore, our
sky Gaussians are not attributed with spurious albedo and
surface normals.
masks provided in the NeRF-OSR dataset (Rudnev
et al., 2022; Kaleta et al., 2025). We benchmark our
approach against state-of-the-art methods described
in Sec. 2.3.
For each scene, we use the camera parameters and
SfM point cloud provided by (Kaleta et al., 2025). We
reproduced the results for LumiGauss (Kaleta et al.,
2025) and NeRF-OSR (Rudnev et al., 2022).
For
the other baselines, we report the results presented
in their respective papers, which were obtained us-
ing the same evaluation protocol. Our approach and
LumiGauss were trained for 40000 iterations, while
NeRF-OSR was trained for 500000 iterations. We
conducted all the experiments on an NVIDIA A100
GPU with 40 GB. See the Appendix for additional
implementation details.
5.2
Results
We present a comprehensive quantitative evaluation
of our method against the baselines in Table 1. The
table is structured to highlight the trade-off between
reconstruction quality and computational efficiency
across the methods built on NeRF and Gaussian Splat-
ting (GS).
Compared to the NeRF-based approaches, R3GW
attains competitive reconstruction quality: while it
achieves higher average PSNR and SSIM scores than

<!-- page 8 -->
Table 1: Quantitative comparison of our method with the baselines on the NeRF-OSR test scenes. For LumiGauss (Kaleta
et al., 2025) and NeRF-OSR (Rudnev et al., 2022) the ∗denotes the results presented in the papers by the authors. We
group methods into NeRF-based and Gaussian Splatting (GS)-based approaches, as NeRF models typically achieve higher
visual fidelity but are more computationally expensive to train and render. The last section shows the evaluation results of the
ablation studies described in Sec. 5.3.
Method
PSNR↑
SSIM↑
MSE↓
MAE↓
lk2
lwp
st
avg
lk2
lwp
st
avg
lk2
lwp
st
avg
lk2
lwp
st
avg
NeRF-based
NeRF-OSR∗(Rudnev et al., 2022)
18.72
16.65
15.43
16.93
0.468
0.501
0.517
0.495
0.014
0.024
0.029
0.022
0.090
0.114
0.113
0.106
NeRF-OSR (Rudnev et al., 2022)
18.80
15.30
15.59
16.56
0.501
0.514
0.511
0.509
0.011
0.034
0.027
0.024
0.090
0.132
0.111
0.111
SR-TensoRF (Chang et al., 2024)
17.30
16.74
15.63
16.56
0.542
0.653
0.632
0.609
0.021
0.024
0.030
0.025
0.096
0.093
0.111
0.100
FEGR (Wang et al., 2023)
21.53
17.57
17.00
18.70
-
-
-
-
0.007
0.018
0.023
0.016
-
-
-
-
SOL-NeRF (Sun et al., 2023)
21.23
17.58
18.18
19.00
0.749
0.618
0.680
0.682
0.008
0.028
0.019
0.018
-
-
-
-
NeuSky (Gardner et al., 2024)
22.50
18.31
16.66
19.16
-
-
-
-
0.005
0.016
0.023
0.015
-
-
-
-
GS-based
LumiGauss∗(Kaleta et al., 2025)
19.59
18.01
17.02
18.21
0.700
0.778
0.729
0.736
0.012
0.017
0.021
0.017
0.085
0.096
0.107
0.096
LumiGauss (Kaleta et al., 2025)
19.37
18.18
16.89
18.15
0.703
0.774
0.724
0.734
0.012
0.017
0.022
0.017
0.085
0.096
0.111
0.097
LumiGauss w/out shadows (Kaleta et al., 2025)
19.03
17.76
16.34
17.71
0.693
0.761
0.709
0.721
0.013
0.018
0.024
0.018
0.090
0.103
0.122
0.105
Ours
20.18
17.93
16.22
18.11
0.731
0.768
0.718
0.739
0.010
0.018
0.025
0.018
0.078
0.098
0.117
0.098
Ablations
Ours w/out cspecular (Eq. (10))
20.03
17.38
15.35
17.59
0.722
0.760
0.706
0.729
0.011
0.020
0.032
0.021
0.080
0.102
0.128
0.103
Ours w/out Lsky-depth (Eq. (16))
20.08
17.86
16.19
18.04
0.728
0.761
0.717
0.735
0.011
0.019
0.025
0.018
0.079
0.101
0.116
0.099
Ground Truth
Ours
LumiGauss
NeRF-OSR
Ground Truth
Ours
d Truth
Ours
Ours
LumiGauss
LumiGauss
NeRF-OSR
Ground Truth
Ours
Ours
LumiG
Ours
LumiGauss
LumiGauss
NeRF-OSR
Figure 6: Qualitative comparison of rendered novel views and illuminations of the NeRF-OSR test scenes (from top to bottom:
lk2, lwp, st) synthesized by our method, LumiGauss (Kaleta et al., 2025) (shadowed version), and NeRF-OSR (Rudnev et al.,
2022). We show novel views employed for the evaluation, rendered using the ground truth environment maps. The sky is
masked out in all the renderings.
NeRF-OSR (Rudnev et al., 2022) and SR-TensoRF
(Chen et al., 2022a), it remains slightly below NeuSky
(Gardner et al., 2024) in PSNR. Nonetheless, NeuSky
requires about 14 hours of training per scene com-
pared to only 2 hours for R3GW. This illustrates
the efficiency advantage of our 3DGS-based formu-
lation even when exact runtime comparisons are lim-
ited. Notably, our method achieves the highest av-
erage SSIM across all methods that report this met-
ric. Since R3GW does not explicitly model shadows,
we additionally compare it to the unshadowed version
of the GS baseline, LumiGauss (Kaleta et al., 2025).
Our approach outperforms LumiGauss on the scene
lk2 across all evaluation metrics, for both the shad-
owed and unshadowed versions of the method. On
the lwp and st scenes, R3GW achieves performance
on par with or better than that of the unshadowed Lu-
miGauss, with higher SSIM and lower MAE scores.
Averaged across all scenes, our method outperforms
LumiGauss unshadowed. We attribute the remaining
difference with LumiGauss shadowed to the absence
of explicit shadow modeling in our current formula-
tion, highlighting a promising direction for future ex-
tensions.
Fig. 6 demonstrates that R3GW reconstructs novel
views of the scenes with high fidelity, including ar-

<!-- page 9 -->
chitectural details, while reproducing realistic novel
illuminations. Notably, all methods render colors that
deviate from the ground truth, as reflected in the rel-
atively low PSNR scores.
We attribute this to the
fact that the models do not predict the illumination
for the test views, but they are directly provided with
the ground truth environment map, which is manually
aligned with the scene reconstruction. Fig. 5 further
demonstrates that our method successfully decom-
poses the material and geometry of the scene, predict-
ing plausible albedo maps and sharper surface nor-
mals than the baselines. Unlike the baselines, R3GW
computes the diffuse color of the foreground Gaus-
sians with albedo in linear space and gamma-corrects
the sum of the diffuse and specular colors. This re-
sults in different tone and contrast in albedo maps,
which are predicted by blending the albedo attributes
of the foreground Gaussians and clamped to [0,1] for
visualization. We present an additional visual com-
parison of our method with LumiGauss (Kaleta et al.,
2025) and NeRF-OSR (Rudnev et al., 2022) in the
Appendix of this paper.
Fig. 3 compares the depth maps predicted by our
method and LumiGauss (Kaleta et al., 2025) to qual-
itatively assess how our decoupled sky representa-
tion affects depth estimation. Unlike LumiGauss, our
approach prevents sky Gaussians from being placed
close to the camera position, which reduces depth ar-
tifacts. The insets in Fig. 4 show that this results in
sharper and halo-free rendering of the foreground out-
lines.
Figure 7: Novel views of the NeRF-OSR scenes relighted
under a novel HDR environment map. The first and second
row show renderings with two different sun orientations,
produced by projecting onto fourth-order SH the same en-
vironment map and rotating it around the y-axis (top-left
corner of each row).
5.3
Ablation Studies
We evaluate the contribution of the specular compo-
nent cspecular (Eq. (10)) in the foreground Gaussians’
color attribute through an ablation study.
To iso-
late its effect, we trained the model assuming purely
diffuse reflections by disabling cspecular and reducing
Training Ground Truth
Reconstructed View and Light
Relighted Novel View
Figure 8: Novel views of the NeRF-OSR scenes relighted
with trained environment maps (top-left corner of the sec-
ond column).
the degree of the environmental light SH to 2. Ta-
ble 1 compares our model to its ablated version. For
all the scenes, the view-dependent specular color en-
hances our model’s ability to synthesize novel views
and illuminations. Importantly, removing the specu-
lar view-dependent component from the color of the
foreground Gaussians results in a drop in both PSNR
and SSIM.
We
further
ablate
the
loss
term Lsky-depth
(Eq. (16)).
The comparison with the full R3GW
model in Table 1 indicates that Lsky-depth improves
both PSNR and SSIM scores.
5.4
Relighting
To demonstrate our model’s relighting capabilities,
we render novel views of the NeRF-OSR scenes us-
ing a novel HDR environment map as illumination,
see Fig. 7.
During training, our method predicts
an environment map for each image. We use these
learned environment maps to relight novel views of
the scenes. At training time, R3GW also predicts the
color of the sky per image. Thus, to illustrate one way
of employing the sky Gaussians at inference, we ren-
der them using the trained sky color. Fig. 8 shows
that the learned illuminations remain consistent when
changing the scene viewpoint.
6
LIMITATIONS AND FUTURE
WORK
Our method does not explicitly model cast shadows or
indirect lighting effects. Thus, incorporating a dedi-
cated shadow modeling component presents a promis-
ing opportunity to further enhance realism and to dis-
entangle surface material appearance and incident il-

<!-- page 10 -->
lumination in the presence of prominent shadows. A
natural extension of our work would be to augment
the environment light with a sun directional light and
adapt shadow mapping techniques to the 3DGS set-
ting. Introducing a directional light could also address
our method’s current limitation in reproducing sharp
specular highlights, which can occur under clear sky
or sunny conditions and are challenging to capture
with SH-based illumination only. Furthermore, since
the sky is part of the distant lighting illuminating the
scene, we believe that relating its appearance to the
environment light could further enhance the relight-
ing capabilities of our approach.
7
CONCLUSIONS
We have presented R3GW, a new relighting pipeline
for Gaussian Splatting that is capable of handling the
complexities of outdoor scenes acquired in uncon-
strained environments, bridging the gap between real-
world capture and controllable rendering. The fore-
ground Gaussians enhanced with a PBR color enable
relighting and view-dependent appearance modeling
of the scene foreground, while the sky Gaussians, ac-
counting for the reconstruction of the sky, preserve
the physical plausibility of the learned scene repre-
sentation and boost the accuracy of the recovered ge-
ometry. The experimental results have shown that our
approach performs comparably to the baselines and
accurately reproduces novel illuminations along with
novel views.
ACKNOWLEDGMENTS
This work has partly been funded by the Euro-
pean Commission (Horizon Europe projects Lumi-
nous, grant no. 101135724 and Spirit, grant no.
101070672).
REFERENCES
Burley, B. (2012).
Physically Based Shading at Disney.
In ACM SIGGRAPH 2012 Courses: Physically Based
Shading in Theory and Practice.
Chang, Y., Kim, Y., Seo, S., Yi, J., and Kwak, N. (2024).
Fast Sun-aligned Outdoor Scene Relighting based on
TensoRF. In Proc. of IEEE/CVF Winter Conference on
Applications of Computer Vision, pages 3626–3636.
Chen, A., Xu, Z., Geiger, A., Yu, J., and Su, H. (2022a).
TensoRF: Tensorial Radiance Fields. In Proc. of Eu-
ropean Conference on Computer Vision, pages 333–
350.
Chen, H., Li, C., and Lee, G. H. (2023). NeuSG: Neural Im-
plicit Surface Reconstruction with 3D Gaussian Splat-
ting Guidance. arXiv preprint arXiv:2312.00846.
Chen, H., Lin, Z., and Zhang, J. (2024). GI-GS: Global
Illumination Decomposition on Gaussian Splatting for
Inverse Rendering. arXiv preprint arXiv:2410.02619.
Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., and
Qiao, Y. (2022b).
Vision Transformer Adapter for
Dense Predictions. arXiv preprint arXiv:2205.08534.
Chung, M. K. (2005). Gauss-Weierstrass Kernel Smooth-
ing on Unit Sphere. Technical report, University of
Wisconsin-Madison.
Cook, R. L. and Torrance, K. E. (1982).
A Reflectance
Model for Computer Graphics.
ACM Trans. on
Graphics, 1(1):7–24.
Gao, J., Gu, C., Lin, Y., Zhu, H., Cao, X., Zhang, L., and
Yao, Y. (2024). Relightable 3D Gaussian: Real-Time
Point Cloud Relighting with BRDF Decomposition
and Ray Tracing. In Proc. of European Conference
on Computer Vision, pages 73–89.
Gardner, J. A., Kashin, E., Egger, B., and Smith, W. A.
(2024).
The Sky’s the Limit: Relightable Outdoor
Scenes via a Sky-Pixel Constrained Illumination Prior
and Outside-In Visibility. In Proc. of European Con-
ference on Computer Vision, pages 126–143.
Grubert, G., Barthel, F., Hilsmann, A., and Eisert, P. (2025).
Improving Adaptive Density Control for 3D Gaussian
Splatting.
In Proc. of the International Conference
on Computer Vision Theory and Applications, pages
610–621.
Haiyang, B., Jiaqi, Z., Songru, J., Wei, H., Tao, L., Yuanqi,
L., Jie, G., Runze, F., Yanwen, G., and Lijun, C.
(2025).
GaRe: Relightable 3D Gaussian Splatting
for Outdoor Scenes from Unconstrained Photo Col-
lections. In Proc. of the IEEE/CVF International Con-
ference on Computer Vision.
Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S.
(2024). 2D Gaussian Splatting for Geometrically Ac-
curate Radiance Fields.
In ACM SIGGRAPH 2024
Conference Papers, pages 1–11.
Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., and
Ma, Y. (2024). GaussianShader: 3D Gaussian Splat-
ting with Shading Functions for Reflective Surfaces.
In Proc. of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 5322–5332.
Kajiya, J. T. (1986). The Rendering Equation. In Proc.
13th annual Conference on Computer Graphics and
Interactive Techniques, pages 143–150.
Kaleta, J., Kania, K., Trzci´nski, T., and Kowalski, M.
(2025). LumiGauss: Relightable Gaussian Splatting
in the Wild. In Proc. of IEEE/CVF Winter Conference
on Applications of Computer Vision, pages 1–10.
Karis, B. (2013). Real Shading in Unreal Engine 4. In ACM
SIGGRAPH 2013 Courses: Physically Based Shading
in Theory and Practice.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G.
(2023).
3D Gaussian Splatting for Real-Time Ra-
diance Field Rendering.
ACM Trans. on Graphics,
42(4):1–14.

<!-- page 11 -->
Kingma,
D.
P.
and
Ba,
J.
(2014).
Adam:
A
Method for Stochastic Optimization. arXiv preprint
arXiv:1412.6980.
Kulhanek, J., Peng, S., Kukelova, Z., Pollefeys, M., and Sat-
tler, T. (2024). Wildgaussians: 3D Gaussian Splatting
in the Wild. arXiv preprint arXiv:2407.08447.
Liang, Z., Zhang, Q., Feng, Y., Shan, Y., and Jia, K. (2024).
GS-IR: 3D Gaussian Splatting for Inverse Rendering.
In Proc. of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21644–21653.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. (2020). NeRF: Rep-
resenting Scenes as Neural Radiance Fields for View
Synthesis.
Communications of the ACM, 65(1):99–
106.
Philip, J., Gharbi, M., Zhou, T., Efros, A. A., and Drettakis,
G. (2019). Multi-view Relighting Using a Geometry-
Aware Network. ACM Trans. on Graphics, 38(4):78–
1.
Ramamoorthi, R. and Hanrahan, P. (2001).
An Efficient
Representation for Irradiance Environment Maps. In
Proc. 28th Annual Conference on Computer Graphics
and Interactive Techniques, pages 497–500.
Rudnev, V., Elgharib, M., Smith, W., Liu, L., Golyanik, V.,
and Theobalt, C. (2022). NeRF for Outdoor Scene
Relighting. In Proc. of European Conference on Com-
puter Vision, pages 615–631.
Schonberger, J. L. and Frahm, J.-M. (2016).
Structure-
from-Motion Revisited.
In Proc. of the IEEE/CVF
Conference on Computer Vision and Pattern Recog-
nition, pages 4104–4113.
Sun, J.-M., Wu, T., Yang, Y.-L., Lai, Y.-K., and Gao, L.
(2023). SOL-NeRF: Sunlight Modeling for Outdoor
Scene Decomposition and Relighting. In SIGGRAPH
Asia 2023 Conference Papers, pages 1–11.
Wang, Z., Shen, T., Gao, J., Huang, S., Munkberg, J., Has-
selgren, J., Gojcic, Z., Chen, W., and Fidler, S. (2023).
Neural Fields meet Explicit Geometric Representa-
tions for Inverse Rendering of Urban Scenes. In Proc.
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8370–8380.
Wu, Z., Chen, J., Li, L., Wu, S., Zhu, Z., Xu, K., Oswald,
M. R., and Song, J. (2025). 3D Gaussian Inverse Ren-
dering with Approximated Global Illumination. arXiv
preprint arXiv:2504.01358.
Xu, C., Kerr, J., and Kanazawa, A. (2024). Splatfacto-W:
A Nerfstudio Implementation of Gaussian Splatting
for Unconstrained Photo Collections. arXiv preprint
arXiv:2407.12306.
Yu, Y., Meka, A., Elgharib, M., Seidel, H.-P., Theobalt, C.,
and Smith, W. A. (2020).
Self-supervised Outdoor
Scene Relighting. In Proc. of European Conference
on Computer Vision, pages 84–101.
Zhang, D., Wang, C., Wang, W., Li, P., Qin, M., and Wang,
H. (2024). Gaussian in the Wild: 3D Gaussian Splat-
ting for Unconstrained Image Collections. In Proc.
of European Conference on Computer Vision, pages
341–359.
Figure 9: Initial positions of the foreground and sky Gaus-
sians. The foreground SfM point cloud is augmented with
the sky Gaussians initial positions sampled uniformly on the
background of the upper hemisphere surrounding the scene.
APPENDIX
Additional Implementation Details
The implementation of our model builds upon the
codebase of 3DGS (Kerbl et al., 2023).
MLP Architecture and Appearance Embeddings.
The MLP first processes the input image embedding
with 3 fully connected layers, each with 256 units.
The output feature vector is then fed into a dense layer
that returns the SH coefficients of the sky color and
into an additional dense layer of size 128. The latter
is followed by a fully connected layer that predicts the
SH coefficients of the environment light. The image
appearance embeddings have dimension 128.
Training Details.
We optimized the parameters of
our model using Adam optimizer (Kingma and Ba,
2014). We set the learning rate for the appearance em-
beddings, MLP weights, and foreground Gaussians’
roughness to 0.0002, to 0.002 for the foreground
Gaussians’ albedo, and to 0.0001 for the radius of the
sky Gaussians. The sky Gaussians’ angle attributes
are optimized using the same learning rate as for the
positions of the foreground Gaussians. The training
starts with 500 warm-up iterations, where only Lfg-sky
is active. Afterwards, all the losses are applied, except
for Lnormal and Lscale, which are started after 2000 it-
erations. We set γ in Lsky-depth to 0.02. Occluders (e.g.
people, cars), which characterize a scene captured in
the wild in addition to its varying lighting, are masked
out using binary segmentation masks in all the losses.
During training, both foreground and sky Gaus-
sians are adaptively controlled as in 3DGS (Kerbl
et al., 2023) (see Fig. 9 for a visualization of the ini-
tial sets of sky and foreground Gaussians). We mod-
ify the original densification scheme by adopting an
exponential ascending gradient threshold to improve
stability during optimization (Grubert et al., 2025).
When a sky Gaussian is split, the sampled positions of
the two children are first projected onto the sky hemi-
sphere. The Cartesian coordinates are then converted

<!-- page 12 -->
Ground Truth
Ours
LumiGauss
NeRF-OSR
Figure 10: Rendered novel views and illuminations of the NeRF-OSR test scenes (from top to bottom: lk2, lwp, st) synthesized
by our method, LumiGauss (Kaleta et al., 2025) (shadowed version), and NeRF-OSR (Rudnev et al., 2022). We show novel
views employed for the evaluation, rendered using ground truth environment maps. The sky is masked out in all the renderings.
Albedo
Surface Normals
Ours
LumiGauss
NeRF-OSR
Figure 11: Additional qualitative comparison of the albedo
map and surface normals predicted by our method, NeRF-
OSR (Rudnev et al., 2022), and LumiGauss (Kaleta et al.,
2025) for the NeRF-OSR scene lk2. The ground truth novel
view is visualized in Fig. 10.
into spherical coordinates to obtain the new polar and
azimuthal angle attributes.
Segmentation Masks.
Our method takes as input
occluders and sky masks, which are required for the
loss computation. We used the strategy of (Gardner
et al., 2024) and extracted them from the segmenta-
tion masks generated using (Chen et al., 2022b).
Additional Qualitative Results
Fig. 10 and Fig. 11 provide an additional quali-
tative comparison of our method with LumiGauss
(Kaleta et al., 2025) and NeRF-OSR (Rudnev et al.,
2022). Fig. 12 shows how the appearance of the scene
changes with viewpoint under fixed lighting.
Figure 12: Comparison of two novel views of the NeRF-
OSR scene lwp rendered under the same trained lighting and
sky color. Insets highlight that the appearance of reflective
surfaces, such as windows, varies with viewing direction.
