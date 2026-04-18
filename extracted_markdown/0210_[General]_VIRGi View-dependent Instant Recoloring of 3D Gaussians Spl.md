<!-- page 1 -->
1
VIRGi: View-dependent Instant Recoloring of 3D
Gaussians Splats
Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez,
Francesc Moreno-Noguer
Abstract—3D Gaussian Splatting (3DGS) has recently trans-
formed the fields of novel view synthesis and 3D reconstruction
due to its ability to accurately model complex 3D scenes and
its unprecedented rendering performance. However, a significant
challenge persists: the absence of an efficient and photorealistic
method for editing the appearance of the scene’s content. In this
paper we introduce VIRGi, a novel approach for rapidly editing
the color of scenes modeled by 3DGS while preserving view-
dependent effects such as specular highlights. Key to our method
are a novel architecture that separates color into diffuse and view-
dependent components, and a multi-view training strategy that
integrates image patches from multiple viewpoints. Improving
over the conventional single-view batch training, our 3DGS
representation provides more accurate reconstruction and serves
as a solid representation for the recoloring task. For 3DGS
recoloring, we then introduce a rapid scheme requiring only
one manually edited image of the scene from the end-user. By
fine-tuning the weights of a single MLP, alongside a module for
single-shot segmentation of the editable area, the color edits are
seamlessly propagated to the entire scene in just two seconds,
facilitating real-time interaction and providing control over the
strength of the view-dependent effects. An exhaustive validation
on diverse datasets demonstrates significant quantitative and
qualitative advancements over competitors based on Neural
Radiance Fields representations.
Index Terms—Neural Radiance Fields, 3D Gaussian Splatting,
Recoloring, Editing, Multi-view Consistency
I. INTRODUCTION
3D Gaussian Splatting (3DGS) [1] has recently established
itself as a dominant technique in the fields of 3D recon-
struction and image-based rendering. By employing a large
set of anisotropic Gaussians to represent scenes, 3DGS fa-
cilitates rapid reconstruction and rendering processes, effec-
tively preserving fine details and complex lighting effects.
Compared to previous neural rendering methodologies such as
NeRFs [2], 3DGS demonstrates markedly superior efficiency
A. Mazzucchelli is with Arquimea Research Center, San Crist´obal de
la Laguna, Santa Cruz de Tenerife, 38320 and Universidad Polit´ecnica de
Catalunya, Doctoral Degree in Automatic Control, Robotics and Vision, Carrer
de Jordi Girona, 31, Les Corts, Barcelona, 08034, Spain.
I. Ojeda-Martin is with Arquimea Research Center, San Crist´obal de la
Laguna, Santa Cruz de Tenerife, 38320.
F. Rivas-Manzaneque is with Volinga AI, San Crist´obal de la Laguna, Santa
Cruz de Tenerife, 38320 and Universidad Polit´ecnica de Madrid, Programa
de Doctorado en Autom´atica y Rob´otica, Calle de Jos´e Guti´errez Abascal 2,
Madrid, 28006, Spain.
E. Garces is with Universidad Rey Juan Carlos, C/Tulip´an S/N, 28933
M´ostoles (Madrid), Spain
A. Penate-Sanchez is with IUSANI, Universidad de Las Palmas de Gran
Canaria, C/ Juan de Quesada 30, Las Palmas de Gran Canaria, 35001, Spain.
F. Moreno-Noguer is with Institut de Rob`otica i Inform`atica Industrial (IRI),
CSIC-UPC, C/ Llorens i Artigas 4, Barcelona, 08028, Spain.
in both reconstruction and rendering. 3DGS has application in
various downstream tasks, including 3D scene/object manip-
ulation [3]–[5], instruction-based editing [6], [7], drag-based
editing [8], multi-object interaction [9] and style transfer [10],
[11]. However, unlike NeRFs, which has been extensively
studied for its recoloring capabilities [12]–[16], the recoloring
problem has received little attention in the context of 3DGS.
In this paper we introduce VIRGi, the first approach for
photorealistic appearance editing of scenes modeled with
3DGS. Our approach is highly efficient and user friendly.
Given a scene modeled with 3DGS, the user simply provides
color edits in a single view. These edits are then rapidly
propagated throughout the entire scene and viewpoints within
approximately two seconds. This, combined with the inher-
ent rendering efficiency of 3DGS, enables a quasi-real-time
interactive editing experience.
Our solution is built upon two primary components. First,
we present a novel architecture and training scheme tailored
for accurately modeling viewpoint dependencies. Traditional
3DGS systems [1] are trained on a per-view basis, with train-
ing batches composed of image tiles from a single viewpoint.
In our approach we implement multi-view training by modify-
ing the CUDA implementation of the rasterizer, enabling the
use of different viewpoints in the same training batch. This
approach, as we will show later, facilitates the decomposition
of reflections into diffuse and view-dependent components
(e.g. specularities), encapsulated in two independent MLPs.
The second component of our approach is a pipeline for
efficient scene recoloring. Given a target view with the desired
color edition, we propose a strategy where we only fine-tune
the last layer of the diffuse MLP, in conjunction with a module
designed to perform soft segmentation of the edited region.
This optimization ensures the consistent propagation of color
edits throughout the entire scene, preserving view-dependent
effects and preventing artifacts such as color bleeding com-
monly observed in other approaches. Since this process op-
erates at interactive rates, users can recursively provide new
edited images if required. We validate our recoloring method
across multiple datasets and conduct a comparative analysis
with previous NeRF-based methodologies. VIRGi consistently
outperforms these approaches not only in efficiency and inter-
activity but also in quality. We demonstrate that our decoupled
diffuse/view-dependent color modeling enables the creation of
highly photorealistic color editions, while also empowering
the user to control specular effects, such as the strength of
the highlights. Fig. 1, and the supplementary video showcase
examples of the results achieved with our method, consistently
arXiv:2603.02986v1  [cs.CV]  3 Mar 2026

<!-- page 2 -->
2
Original image
Recolor
Multi-recolor
Novel view
Diffuse
Specular
Fig. 1: VIRGi
enables – by editing a single view – instant recolorings of one or multiple objects in scenes modeled with
3D Gaussian Splats. Our key idea of decomposing the reflectance of the scene into diffuse and specular components allows
view-dependent consistent edits. Our separation can also be used to enhance or reduce material specularity.
surpassing competing approaches as demonstrated in Fig. 2.
To summarize, we provide the following contributions:
• The first method for conducting photorealistic, view-
dependent, and interactive color edits for scenes modeled
with Gaussian Splats. Our approach is designed to be user-
friendly and interactive, requiring only a single color edit
from one view to recolor the entire scene.
• A novel neural network architecture for 3DGS that learns the
decoupling of color into its specular and diffuse components.
This architecture enables precise color editing of scenes
while preserving their view-dependent effects.
• A multi-view training strategy that is essential to learning
the diffuse/specular color components. Incorporating infor-
mation from different viewpoints during training enhances
not only the learning of the decomposed output in our novel
architecture, but also the overall performance of the 3DGS
reconstruction.
II. RELATED WORK
3D Gaussian Splatting [1] provides real-time rendering with
simpler manipulation compared to Neural Radiance Fields
(NeRFs) [2] because of its explicit scene representation. How-
ever, globally editing 3DGS scenes remains an open problem,
as we discuss in this section.
Point and 3DGS Rendering. Gaussian Splats originate from
point-cloud representations [17]. These techniques became
very popular in 3D modeling [18], [19] and as the basis
for image-based rendering techniques [20]–[24]. In recent
years, Gaussian Splats [1] have surpassed previous point-based
rendering methods by enabling real-time novel view synthesis
for scenes of varying complexity. These scenes range from
those with simple Lambertian materials to those with intricate
light interactions, dynamic elements [9], [25]–[30], physical
simulators [9], and avatars [31]–[39].
Many works have already addressed the memory and stor-
age limitations of 3DGS, leading to efficient compression
strategies that now allow deployment even on lightweight
devices [40]. In particular, vector clustering and quantization-
based methods have proven effective in minimizing parameter
counts [41]–[44]. Notably, [45] employ vector quantization
to construct codebooks. However, unlike other methods, their
approach explicitly separates color information from geometry.
A major drawback that remains to be addressed, and which
we tackle in this paper, is the constraint of performing the
optimization step over a single image. Unlike NeRF, where
rays from different viewpoints can be sampled with minimal
overhead, the 3DGS rasterization pipeline is inherently built
around a single image. In this work, we adopt the represen-
tation from [45] and extend it by augmenting the CUDA
rasterizer so that optimization can be done from multiple
images in a single step.
Inverse Rendering in Radiance Fields. NeRF-based ap-
proaches have incorporated surface normals and separated
diffuse/specular components to improve reflectance modeling
and enable scene relighting, as demonstrated in [46], [47].
Other works [48]–[50] jointly optimize geometry, materials,
and illumination, leveraging physics-based priors or path trac-
ing with near-field indirect light to better disentangle reflective
properties from environmental lighting. In the domain of 3D
Gaussian Splatting, inverse rendering has been enhanced by
estimating normals, materials, and indirect lighting for each
Gaussian primitive [51]. To better handle ambiguities between
surface normals and roughness, a segmentation prior is intro-
duced in [52], guiding geometry and material decomposition.
Normals are estimated from the shape of each Gaussian
in [53], where directional masking enforces orientation, en-
abling BRDF rendering and disentangling indirect light via
voxel-based tracing. A dedicated specular shading is proposed
in [54] to refine surface normals and enhance high-frequency
reflections without compromising splatting efficiency.
Color Editing. Color editing through edit propagation strate-
gies utilizes local color cues, such as user-defined scribbles
or points, to influence similar regions within the image.
Previous works in this domain employed propagation rules
based on color similarity metrics [55]–[58]. However, defining
effective metrics that capture both local and non-local image
features posed a challenge. To address this, Convolutional
Neural Networks (CNNs) proved successful for robust fea-
ture extraction [59], [60]. A different approach involves pre-
computed palettes representing dominant colors in an image.
These palettes serve as a basis for describing other colors
through linear combinations. Some artists prefer this method
for its control over global and local color modifications, often
combining it with object segmentation techniques [61]–[63].

<!-- page 3 -->
3
Fig. 2: Comparison of recoloring methods with a ground truth recolor. Our method VIRGi is the first one to propose a
solution for 3DGS recoloring, outperforming one of the best methods in NeRFs recoloring [15] (IReNe). We propose several
contributions that make our method robust and better than a baseline (VANILLA-VIRGi) obtained by na¨ıvely combining existing
methods.
In this work, we propose a method that can leverage any of
these paradigms to transition from a 2D to a full 3D edit.
Our method allows users to edit a single 2D image using their
preferred editing tool or technique. Thanks to our efficient
propagation strategy, any 2D recoloring is extrapolated to a
3DGS-based representation.
Radiance Fields Color Editing. Despite their brief existence,
Gaussian Splats have sparked rapid development, with various
manipulation techniques emerging. Text prompts are used to
modify static scenes [6], [64], [65] or dynamic ones [66],
[67]. Additionally,
[68] enable object selection, while [69]
create articulated Gaussian Splats from dynamic sequences.
[5] propose diverse edits, such as semantic segmentation,
object segmentation, or text-based editing, through feature
distillation. Unlike for NeRFs [12]–[14], [16], to the best of
our knowledge, there is no specific work for Gaussian Splat
recoloring. Our work draws inspiration from [15], who, by
fine-tuning part of the NeRF, achieved remarkable quality
and view-dependent instant recoloring. However, as we will
demonstrate in the experimental section, our method, which
incorporates multi-view training and explicit diffuse/specular
modeling, significantly enhances the quality compared to [15].
III. BACKGROUND AND VANILLA-VIRGI
We
ground
our
approach
on
3D
Gaussian
Splatting
(3DGS) [1], particulary focusing on Compact 3D Gaussian
Splatting (C3DGS) [45]. 3DGS represents the radiance field
by generating a collection of 3D Gaussians, each incorporating
spherical harmonics information. During the rendering pro-
cess, these Gaussians are splatted onto a 2D plane to produce
the 2D image [70]. However, a notable limitation of 3DGS is
its significant memory requirement, averaging 746 MB for the
Mip-NeRF dataset [71]. To address this challenge, C3DGS
introduced two solutions: encoding the Gaussians using a
codebook to conserve memory, and a key contribution for
our purposes, utilizing a hashgrid and an MLP to encode
appearance information. Their latter contribution eliminates
the necessity to store spherical harmonics coefficients for
each Gaussian, thereby reducing the memory footprint by a
factor of 15. Consequently, the memory requirements decrease
substantially to an average of 48.8 MB on Mip-NeRF dataset.
Our proposed approach adapts the recoloring strategy for
NeRFs from IReNe [15], one of the top-performing methods
in this domain. IReNe takes as input a single recolored view
of the NeRF and propagates the edits to the entire NeRF
using a strategy based on fine-tuning part of the color MLPs,
selecting the neurons responsible for view-dependent effects,
and performing soft segmentation in 3D space. We leverage
some of these concepts and tailor them to suit C3DGS.
A. VANILLA-VIRGi
Given the differing radiance field representations between
NeRFs and 3DGS, combining both strategies presents chal-
lenges. Our initial idea, dubbed VANILLA-VIRGi, attempted
to integrate the MLP configuration and hashgrid features
from C3DGS, along with the last layer and soft-segmentation
blending from IReNe in a straightforward manner. Specifically,
VANILLA-VIRGi utilizes a single neural network to estimate
the color C
of each Gaussian, following the formulation
proposed in [45], expressed as:
C = MLPc(f , θ|ϕc),
(1)

<!-- page 4 -->
4
3D Gaussians
Image
edited
by user
MLPseg
Cspec
Cdiff
C’diff
Alpha
blending
Recolored scene
R
G
B
G
B
R
C’’diff
MLPdiff
Hashgrid
Multi-view Training
(Section 4.2)
Trained before color edition
Trained during color edition
Dense connections:
Editing (Section 4.3)
C’
hdiff
MLPspec
R
G
B
Diffuse/Specular Decomposition (Section 4.1) 
Input images
C
Fig. 3: Overview of VIRGi. Our method comprises two main steps. In the initial step, we train a 3DGS architecture consisting
of a Hashgrid f representing the geometry and two MLPs for color modeling. We separate the observed color into a diffuse
term MLPdiff, which solely relies on hashgrid features f, and a specular term MLPspec, which incorporates the view direction
θ. Our training strategy involves leveraging multiple views of the same scene point in a single batch, as opposed to using a
single image per batch, resulting in improved PSNR compared to previous methods. In the second step, given an edited 2D
image, we fine-tune the last layer of the diffuse MLP using a soft-segmentation mask α to achieve the target color.
where f are the hashgrid features, θ denotes the view direction,
and ϕc are the weights of the MLP. Then, following the
approach outlined in IReNe [15], VANILLA-VIRGi modifies
only the last layer of MLPc when an edit is performed,
resulting in a new color:
C′ = MLPc(f , θ|ϕlast
c
).
(2)
C′ is blended with the original to produce the final color C′′:
C′′ = αC′ + (1 −α)C.
(3)
Blending is performed as α = MLPseg(f ), which is a soft
segmentation of the edited region and is learned by an MLP
from the hashgrid space features. Additionally, this approach
incorporates a view-dependent neuron selection strategy within
the last layer of MLPc. For further details, we encourage
readers to refer to the original paper [15].
However, as depicted in Figure 2, this straightforward ap-
proach does not yield optimal results, as traces of the previous
color persist in the rendering of novel views. This outcome
is somewhat expected, as this method lacks mechanisms to
enforce multi-view consistency, leading to the entanglement
of view-dependent effects such as specular highlights within
the color MLPc.
Our proposed solution, VIRGi, aims to establish a clear
separation between diffuse and view-dependent (specular)
effects [13], [72], [73] from the initial construction of the
Gaussian Splats. We introduce two pivotal concepts detailed
in the following section: Firstly, a decoupled diffuse-specular
color estimation through distinct MLPs; and secondly, a multi-
view training strategy that leverages the principle that the
diffuse color of an object with isotropic reflectance remains
invariant to view changes [74].
IV. METHOD
Figure 3 provides an overview of our method, which operates
in two main steps. First, we initiate the process by constructing
a compact 3D scene representation [45] from a collection of
multi-view images. This learned representation comprises two
primary components: (i) a set of hashgrid features f , which
capture the geometric details of the scene, and (ii) a pair
of Multi-Layer Perceptrons (MLPs) responsible for encoding
the scene’s appearance. Unlike C3DGS, we employ separate
MLPs, MLPdiff and MLPspec, to model diffuse color Cdiff and
specular Cspec reflections, respectively. The architecture of this
decoupling is explained in Section IV-A.
To train our 3DGS, we employ a novel optimized multi-view
training strategy, detailed in Section IV-B, that implicitly en-
courages corresponding 3D points to occupy nearby locations
within the neural feature space. This novel approach, unlike
conventional training with single images per batch [1], [45],
captures more consistent and better reflections, as demon-
strated in the experiments section. Once the original scene
is learned, users can utilize any existing tool to recolor one
view of the scene Iedit. Subsequently, our optimized fine-tuning
process, described in Section IV-C, ensures that the color
edits are consistently propagated throughout the entire scene
at interactive rates.

<!-- page 5 -->
5
Original image
Monoview
Multiview
Fig. 4: Specularity Comparison: Monoview vs. Multiview.
Training with a single viewpoint poses challenges in learning
specularities due to limited sample diversity, multiview train-
ing facilitates an easier understanding of specular effects.
A. Diffuse/Specular Decomposition Architecture
Separating diffuse Cdiff and view-dependent color Cspec (or
specular) in independent MLPs [13], [15] is essential for
ensuring view-dependent consistency in the edits. This sep-
aration reflects the underlying assumption that object color
is primarily captured by the diffuse component which is
reasonable for many common scenarios.
To model the view dependency, MLPdiff only utilizes the
hashgrid features f , whereas MLPspec incorporates both the
hashgrid features and the view direction θ. The outputs of
these components are combined using a sigmoid function σ(·),
expressed formally as:
Cdiff = MLPdiff(f |ϕdf )
(4)
Cspec = MLPspec(f , θ|ϕsp, ϕdf )
(5)
C = σ(Cdiff + Cspec)
(6)
where ϕsp and ϕdf denote the specular and diffuse weights of
the MLPs, respectively, and C represents the RGB color of
the Gaussian.
The standard architecture of MLPspec and MLPdiff consists of
three fully-connected layers. However, we found it necessary
to introduce additional residual connections between them.
Specifically, we connected the first layer of MLPdiff to the sec-
ond layer of MLPspec, and the second layer of MLPdiff to the
third layer of MLPspec. These connections are unidirectional,
allowing only MLPspec to access the content of MLPdiff.
The rationale behind this architecture is as follows: while
MLPspec theoretically accounts for all view-dependent re-
flectance (since it receives θ as input), MLPdiff should focus
solely on learning what is view-point invariant. Without these
connections, the specular component tends to be noisier,
making it challenging to learn an accurate reflection. With
the added connections, the specular component gains a better
understanding of which aspects of the reflection to prioritize.
To train the pair of MLPs we use the following loss function:
L = ∥C −Iedit∥1 + λspec∥Cspec∥2
2
(7)
This loss function uses an L1 photometric loss with a smooth-
ness term on the specular component. The value of λspec grad-
ually decreases from 0.25 to 0.05 during training. Initially, this
penalty encourages the network to prioritize explaining color
variations using the diffuse component (Cdiff) alone. Since the
diffuse MLP (MLPdiff) lacks view direction information, it
cannot effectively capture view-dependent effects. As training
progresses, we reduce the penalty on MLPspec to allow the
network to also account for the view-dependent part.
B. Multi-view Training
An essential aspect of our method is the multi-view training
strategy. Unlike existing approaches in Gaussian Splats [1],
[45], which utilize one image per batch, our method samples
multiple views simultaneously. This strategy offers two key
advantages. Firstly, it encourages points in the 3D world
with similar reflectance properties to be close together in
the feature space, thereby aiding convergence and enhancing
overall quality. Secondly, it assists in implicitly decomposing
the reflection into diffuse and specular components, which
proves advantageous for the recoloring task.
We adapted the CUDA implementation of the fast rasterizer
to use tiles from different images within the same training
step. Additionally, to expedite training time without altering
the geometry during recoloring, we initialize the 3D Gaussians
using C3DGS and freeze them. This ensures interactivity,
enabling faster operations as there’s no need to recompute the
gaussian ordering for alpha blending during rasterization. With
the precomputed ordering, we create mini-batches composed
of five tiles from different images, where the same gaussian
is observed from diverse viewpoints. The training batch is
comprised of as many 5-sample mini-batches as can be ac-
commodated within the GPU’s VRAM.
As we demonstrate in the experimental section, this modifica-
tion alone significantly enhances the default rendering quality
of different baselines (see Table III). Additionally, multi-view
training is essential for effectively learning a robust decompo-
sition of both diffuse and view-dependent components. In the
original 3DGS approach, training is conducted using a single
image per batch, meaning each Gaussian is updated based on
this singular viewpoint. However, by including five samples
of each Gaussian within the same training batch, the update
process endeavors to account for all these viewpoints simulta-
neously. This aids the MLPdiff in discerning the shared color
aspects (diffuse) and the viewpoint-dependent characteristics
(specular) among these five tiles. In Fig. 4, we provide two
examples where the benefits of utilizing multi-view training
for learning the specular components are evident.
C. Editing
The recoloring process involves the user modifying a desig-
nated target view Iedit, followed by fine-tuning a portion of
the architecture to accommodate the desired changes. Drawing
from [15], we implement last-layer adaptation on the diffuse
component (as per Equation 2) and soft-segmentation (as
per Equation 3). We update the weights ϕ′
df of the last
layer of MLPdiff to derive a new diffuse color C′
diff. This
new color is then alpha-blended with Cdiff using the soft-
segmentation mask, denoted as α, resulting in the final diffuse
color denoted C′′
diff. It’s worth noting that our method differs
in the way the soft-segmentation mask is obtained compared
to [15]. In our approach, we establish a connection between the

<!-- page 6 -->
6
Mip NeRF 360 [71]
LLFF [75]
NeRF Synthetic [2]
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PaletteNeRF
22.24
0.720
0.278
24.19
0.800
0.187
29.92
0.960
0.038
RecolorNeRF
-
-
-
24.52
0.736
0.333
29.00
0.943
0.048
IReNe
26.80
0.714
0.292
26.51
0.823
0.136
30.33
0.959
0.035
VANILLA-VIRGi
29.19
0.881
0.150
28.01
0.905
0.129
32.13
0.975
0.023
VIRGi
29.74
0.886
0.146
29.10
0.915
0.119
31.65
0.971
0.027
TABLE I: Quantitative color edition results. Best, Second. Results using the recolored ground truth images from [15].
Mip NeRF 360
LLFF
NeRF Synthetic
DC
MV
DS
PSNR↑
PSNR↑
PSNR↑
✗
✗
✗
29.19
28.01
32.13
✗
✓
✗
29.45
28.54
31.90
✗
✓
✓
29.56
28.62
31.78
✓
✗
✗
28.58
27.66
30.93
✓
✓
✗
29.61
29.15
31.50
✓
✓
✓
29.74
29.10
31.65
TABLE II: Ablation study. Recoloring results on all datasets
of our ablated method. Best, Second. DC indicates if the output
is decoupled or not. MV indicates if the model has been trained
using our multi-view rasterization DS indicates if the diffuse’s net-
work MLPdiff output has been introduced into the soft segmentation
network MLPseg.
activations of last layer of the diffuse MLP, denoted as hdiff,
and the segmentation MLP. This integration aims to introduce
additional contextual information to improve the quality of
segmentation. As demonstrated in the experiments section,
this enhancement contributes to the overall performance of
our approach.
α = MLPseg(f , hdiff),
(8)
C′
diff = MLPdiff(f |ϕ′
df ),
(9)
C′′
diff = αC′
diff + (1 −α)Cdiff,
(10)
C′ = σ(C′′
diff + Cspec)
(11)
Upon receiving the user’s edit, we fine-tune the color and
segmentation MLPs (MLPdiff, MLPspec, and MLPseg) to
accommodate the desired changes with the following loss
function:
L = ∥C′ −Iedit∥1
(12)
This loss function uses an L1 photometric loss between the
rendered recolored version C′ and the user’s edit Iedit.
V. RESULTS
A. Quantitative results
Comparison with state-of-the-art. In Table I, we present
the results of recoloring on the dataset provided by [15],
which includes manually recolored scenes using photoshop
in LLFF [75], MipNeRF-360 [71], and recolored scenes from
Synthetic NeRF [2] using Blender. The original code from
[1] did not converge in the flower and leafs scenes from
LLFF, so those scenes were not evaluated for any of the re-
ported methods. We compare our results against several recent
baselines, including PaletteNeRF [13], RecolorNeRF [12], and
IReNe [15]. We provide results from our proposed method,
VIRGi, as well as from the VANILLA-VIRGi. In Fig. 8 some
qualitative results from the dataset can be seen. The evaluation
Dataset (PSNR↑)
Mip NeRF 360
LLFF
NeRF Synthetic
Mono-view C3DGS
28.96
25.63
35.60
Multi-view C3DGS
29.47
25.87
35.84
Mono-view VIRGi
28.43
25.73
33.87
Multi-view VIRGi
29.30
25.97
35.76
TABLE III: Multi-View VS Mono-View novel view synthe-
sis. Training both the original C3DGS as well as VIRGi using
the proposed multi-view approach enhances the novel view
synthesis results. We underline that VIRGi also provides a
decomposition of the image into diffuse and specular.
metrics employed are standard radiance field metrics (PSNR,
SSIM, and LPIPS), and the split between training and testing,
as well as the recoloring editions, follows the protocol estab-
lished in [15]. Importantly, VIRGi demonstrates a notable and
consistent improvement over other radiance field recoloring
methods, underscoring its efficacy and robustness.
Method robustness. The dataset provided by [15] contains
11 distinct recolored images for each scene in the MipNeRF-
360 and LLFF datasets. To demonstrate the robustness of
our approach to variations in the input image, we conducted
11 evaluations for each scene, each time using a different
recolored ground-truth image as input. This methodology
allows us to assess that the quality of our method remains
consistent regardless of the specific input image chosen by the
user for editing. The average recoloring PSNR for 11 iterations
when modifying the edited image for VIRGi was (MipNeRF-
360: 29.27, LLFF: 28.87), while for VANILLA-VIRGi it was
(MipNeRF-360: 28.71, LLFF: 27.91). Remarkably, the aver-
age performance in both cases closely aligns with the standard
experiment in Table I, using the images defined by [15] as the
input image.
Ablation study. We analyze the individual contribution of
each component with two ablation experiments. The first
experiment (Table II), analyzes the impact of the decoupled
output (DC), the multi-view rasterization (MV), and the in-
tegration of diffuse neuron activation into the segmentation
network (DS). Notably, our method VIRGi emerges as the
best-performing recoloring method overall. Adding multi-
view information (MV) consistently enhances performance
across all configurations. Moreover, the decoupled architec-
ture (DC) demonstrates superior results with the utilization
of multi-view. Remarkably, when multi-view is employed,
the decoupled results consistently outperform the original
non-decoupled color estimation. Additionally, inputting the
diffuse neuron activations to the segmentation (DS) boosts
performance in all cases except for a singular instance in
the LLFF dataset. This outcome is reasonable, as the LLFF
dataset comprises frontal recordings with limited viewpoint

<!-- page 7 -->
7
Fig. 5: Comparison between VIRGi (upper row per case)
and Gaussian Editor [68] (bottom row per case). Gaussian
Editor requires the user to initialize a mask using SAM [76]
before applying color edits based on provided prompts:
“kitchen - make it magenta”, “bicycle - make it yellow”, and
“bonsai - make it green”. Gaussian Editor exhibits several
drawbacks, such as color bleeding into surrounding areas in
the first two scenes and not understanding the scene structure
when recoloring the third scene. Notably, Gaussian Editor
takes an average of 612 seconds to process these scenes, while
VIRGi completes the task in just 2 seconds on average.
changes, thereby posing challenges in understanding diffuse
and specular aspects of the scene.
In our second ablation experiment (Table III), we demonstrate
how multiview optimization not only enhances the quality of
the recoloring but also inherently improves the modeling of
view-dependent effects in 3DGS. Specifically, we compare the
reconstruction accuracy (without recoloring) of the original
C3DGS [45] trained with and without multiview, and apply
the same procedure to evaluate VIRGi. Notably, multiview
consistently yields performance improvements across all cases.
Furthermore, this experiment confirms that the proposed mod-
ifications to C3DGS [45] for color editing do not lead to any
noticeable reduction in the original quality.
B. Qualitative results
Fig. 7 showcases qualitative results obtained from the
MipNeRF-360 dataset [71]. This dataset features complex
real scenes with complete 360° coverage. The edits produced
by VIRGi
demonstrate consistency across viewpoints and
Fig. 6: Illumination editing with decoupled output. Top
row: specular component reduced by applying a multiplicative
factor of 0. Bottom row: specular component increased by
applying a factor of 1.5.
effectively capture the intricate view-dependent effects present
in the scenes. Additionally, the 3D segmentation masks for
each recolored region, estimated by our method, are presented,
further illustrating its capability to achieve 3D consistency.
Comparison with Gaussian Editor. As discussed in Sect. II,
no other method specifically addresses the challenge of per-
forming appearance edition on Gaussian Splatting. However,
some methods enable the modification of the final color
of specific elements using text prompts. Fig. 5 presents a
qualitative comparison of VIRGi and GaussianEditor [68].
GaussianEditor facilitates single-element recoloring by lever-
aging SAM [76] to generate a mask of the desired object.
Subsequently, guided by a text prompt and employing Instruct-
Pix2Pix [77], the scene edit is optimized until satisfactory re-
sults are achieved. As shown, our method is more consistently
more robust. Notably, GaussianEditor exhibits color bleeding
into undesired areas of the scene, a limitation not observed in
VIRGi’s results. Additionally, while GaussianEditor requires
approximately 10 minutes to complete a given edit, our
approach achieves the same task in just 2 seconds, significantly
enhancing its interactivity and usability.
Specular editing. In Fig. 6, we illustrate how decoupling the
color output into diffuse and specular components enables fine
adjustments to view-dependent effects. By manipulating the
strength of the specular component Cspec, we can make subtle
changes in the material appearance. Although the range of
editions is limited, this experiment provides valuable insights
into our model’s understanding of scene appearance, a key
aspect of its superiority over other recoloring methods.
C. Limitations
While our approach excels in handling a wide range of color
editing tasks, it has several limitations. First, our current
material model with diffuse/specular separation is very simple,
thus, more complex materials with transparency, anisotropy,
or with colored reflections are not captured by the proposed
representation. In Fig. 9, first and second rows, we show two
examples of our decomposition. In the scene named as Kitchen
(first row, the one with the Lego), the result appears more
coherent and smooth, successfully separating reflections from
the table and the top of the tractor wheels. While the Kitchen

<!-- page 8 -->
8
scene handles reflections reasonably well, we observe that the
decomposition breaks down in more complex cases, such as
the mirror-like surfaces in the Counter scene (second row).
As discussed previously, the goal of our decomposition is not
physical accuracy but editability: specifically, enabling stable
and multiview-consistent recoloring without degrading the
final render. This is precisely why we introduced the separation
of MLPs in our design: to ensure that recoloring operations
target only the components unaffected by view direction,
maintaining consistency across novel views. To show what
our method would yield as a result in such failure cases, we
have performed in Fig 9 some examples where we try to edit
a transparent object, the water filter jug, and an object with
a mirror-like surface, the bowl. In both cases our approach
edits parts of the scene that it should not modify, generating
a bleeding out of the color edit into other objects. Another
limitation is that in certain cases a single edited view does not
have enough information to segment the scene perfectly. This
happens mainly in cases in which some parts of the scene,
that were not present in the original edited image, are fairly
similar to parts of the desired edition. Luckily, due to our
method being interactive this can be easily fixed by adding a
second edited image, as we show in the supplementary video.
We show examples of wrong recoloring cases in Fig. 10. In
those cases we show how we tried to solve the issues by adding
a second edited image.
VI. CONCLUSIONS
We have presented a novel method that, by disentangling
the color components and integrating multiview information
during training, achieves high-quality in color editing for
3DGS. This represents a significant advancement, marking the
first instance of a Gaussian splatting editing technique capable
of generating recolored models of superior quality. Moreover,
our method operates with remarkable speed (∼2 seconds),
rendering it highly suitable for interactive use by virtual
production engineers or game designers seeking to efficiently
edit assets. As future work, we aim to explore more complex
materials and appearances, such as extending to specular
color editing. Furthermore, investigating the minimum number
of views required to ensure consistent representations while
reducing memory consumption for complex edits is another
promising direction.

<!-- page 9 -->
9
Original image
Ground truth
recolor
VIRGi
segmentation
VIRGi edition
VIRGi
segmentation
VIRGi edition
Fig. 7: Qualitative results. Results on all scenes from the MipNeRF-360 dataset [71]. This dataset shows complex real scenes
with a complete 360 rotation on all scenes. We show several examples of single and multiple color editions. The first column
is the original image. The second column is the edit performed by the user and used to train VIRGi. We also show the soft
segmentation predicted by VIRGi for each recolored image.

<!-- page 10 -->
10
1. Unedited GT
2. Edited GT
3. IReNe
4. VIRGi
Fig. 8: Some qualitative results of the IReNe [15] dataset.

<!-- page 11 -->
11
Ground Truth
VIRGi Diffuse
VIRGi Specular
VIRGi Combined RGB
Ground Truth
Edit
VIRGi Edited pose render
VIRGi Novel View render
Fig. 9: Limitations of diffuse/specular modeling. First row Kitchen scene, second row Counter scene. Comparison between
Ground Truth, Diffuse, Specular and Combined RGB render. Our simple diffuse/specular model struggles with complex or
anisotropic materials. This can be seen in the Counter (bottom) scene, while the Kitchen (top) scene, especially the table,
shows better results. Still, both yield solid RGB reconstructions. Third row - Fourth Row: Comparison of original, edited, and
rendered views on specular failure cases. In the case of the water jug, although we are able to apply the recoloring, the result
lacks uniformity: some areas of the jug appear more red than others, and the color bleeds onto objects visible through the
jug’s transparent surface. For the reflective bowl, we manage to change its color as well, but the edits are less clean compared
to those on non-reflective, opaque objects. Some regions of the bowl retain the original color, with only a slight green tint
appearing in the edited output

<!-- page 12 -->
12
Fig. 10: Limitations in segmentation. We show examples where issues in recoloring happened. These cases occur mainly
with parts of the scene that were not present in the first edited image. In the first example, the notebook has been incorrectly
recolored. In the second example, the palm tree has been incorrectly recolored. In the third example, the legs of the table have
not been fully recolored to red. These cases were fixed by adding a second edited image. In most cases, such issues can be
solved by adding an additional edited image, but in some cases, like the last example, certain edits cannot be achieved. In
this particular case, the curtain is far from the camera, and not much viewpoint change is present in the dataset from which
to learn the correct model.

<!-- page 13 -->
13
Alessio Mazzucchelli received the B.S. degree and
the M.S. degree in computer engineering at Po-
litecnico di Milano. He is currently pursuing the
Ph.D. degree at Universitat Polit`ecnica de Catalunya
while developing its research at Arquimea Research
Center. His research is mainly focused on editing
implicit and explicit 3d representation.
Ivan Ojeda-Martin received the B.S. degree in
computer engineering at Universidad de Las Palmas
de Gran Canaria. He is currently developing research
at Arquimea Research Center.
Fernando Rivas-Manzaneque received the B.S.
degree and the M.S. degree in electrical engineering
at Carlos III University of Madrid. He is currently
pursuing the Ph.D. degree with Technical University
of Madrid while developing its research at Volinga
AI. His research is mainly focused on neural fields
and his research interests include neural rendering,
scene reconstruction, and image processing.
Elena Garces is a Senior Research Scientist at
Adobe Paris since June 2024. Before that, she was
Juan de la Cierva Fellow at the MSLab of the URJC
and a Director at SEDDI (Madrid) where she led
a technology area with expertise in render, optical
capture, and AI. During 2016-2018, she was a Post-
doctoral researcher at Technicolor R&D (France).
She completed her Ph.D. in 2016 in the Graphics and
Imaging Lab of the University of Zaragoza (Spain).
During her Ph.D. she had the opportunity to intern
twice at Adobe Research (San Jose and Seattle,
US). She proudly received the 2017 Premio de Investigaci´on de la Sociedad
Cient´ıfica Inform´atica de Espa˜na y Fundaci´on BBVA. Press coverage: FBBVA,
Innova Spain, El Pa´ıs, El Mundo El Espa˜nol. Her research interests span the
fields of computer graphics, computer vision, and applied machine learning.
In particular, I currently work on digitization (optics&physics) of fabric
materials, machine learning-based simulations of garments, and 3D scene
reconstruction.
Adri´an Penate-Sanchez (PhD) received his PhD
from
the
CSIC-UPC
robotics
lab
(Institut
de
Rob`otica i Inform`atica Industrial) in Barcelona un-
der the supervision of Juan Andrade Cetto and
Francesc Moreno Noguer. In 2014, he joined the
Computer Vision group at Toshiba’s Cambridge Re-
search Laboratory for a 5 month internship. From
Nov 2015 till February of 2018 he was a Post-
doctoral Research Associate at University College
London (UCL) working in the fields of 3D computer
vision and machine learning. In March of 2018 he
joined the University of Oxford as a Postdoctoral Research Associate working
to find solutions to robot navigation in unstructured environments like forests.
In 2020 he joined the University of Las Palmas de Gran Canaria as a
Distinguished Researcher and Lecturer under a Beatriz Galindo grant.
Francesc Moreno-Noguer is a Research Scientist
of the Spanish National Research Council at the
Institut de Robotica i Informatica Industrial. His re-
search interests are in Computer Vision and Machine
Learning, with topics including human shape and
motion estimation, 3D reconstruction of rigid and
nonrigid objects and camera calibration. He received
the Polytechnic University of Catalonia’s Doctoral
Dissertation Extraordinary Award, several best paper
awards (e.g. ECCV 2018 Honorable mention, ICCV
2017 workshop in Fashion, Intl. Conf. on Machine
Vision applications 2016), outstanding reviewer awards at ECCV 2012 and
CVPR 2014, 2021, and Google and Amazon Faculty Research Awards in
2017 and 2019, respectively. He has (co)authored over 150 publications in
refereed journals and conferences (including 10 IEEE Transactions on PAMI,
5 Intl. Journal of Computer Vision, 28 CVPR, 11 ECCV and 9 ICCV)
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions
on Graphics (TOG), vol. 42, no. 4, July 2023. [Online]. Available:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” in Proceedings of the European Conference on Computer
Vision (ECCV), 2020.
[3] Y. Wang, Q. Wu, G. Zhang, and D. Xu, “Gscream: Learning 3d geometry
and feature consistent gaussian splatting for object removal,” arXiv
preprint arXiv:2404.13679, 2024.
[4] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 5354–5363.
[5] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You,
Z. Wang, and A. Kadambi, “Feature 3dgs: Supercharging 3d gaussian
splatting to enable distilled feature fields,” in The IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2024.
[6] J. Fang, J. Wang, X. Zhang, L. Xie, and Q. Tian, “Gaussianeditor:
Editing 3d gaussians delicately with text instructions,” in The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[7] J. Wu, J.-W. Bian, X. Li, G. Wang, I. Reid, P. Torr, and V. Prisacariu,
“GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting
Editing,” ECCV, 2024.
[8] S. Shen, J. Xu, Y. Yuan, X. Yang, Q. Shen, and X. Wang, “Draggaussian:
Enabling drag-style manipulation on 3d gaussian representation,” arXiv
preprint arXiv:2405.05800, 2024.
[9] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang,
“Physgaussian: Physics-integrated 3d gaussians for generative dynam-
ics,” 2024.
[10] K. Liu, F. Zhan, M. Xu, C. Theobalt, L. Shao, and S. Lu, “Stylegaus-
sian: Instant 3d style transfer with gaussian splatting,” arXiv preprint
arXiv:2403.07807, 2024.
[11] D. Zhang, Z. Chen, Y.-J. Yuan, F.-L. Zhang, Z. He, S. Shan, and L. Gao,
“Stylizedgs: Controllable stylization for 3d gaussian splatting,” arXiv
preprint arXiv:2404.05220, 2024.
[12] B. Gong, Y. Wang, X. Han, and Q. Dou, “Recolornerf: Layer decom-
posed radiance fields for efficient color editing of 3d scenes,” arXiv
preprint arXiv:2301.07958, 2023.
[13] Z. Kuang, F. Luan, S. Bi, Z. Shu, G. Wetzstein, and K. Sunkavalli,
“Palettenerf: Palette-based appearance editing of neural radiance fields,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 20 691–20 700.
[14] J.-H. Lee and D.-S. Kim, “Ice-nerf: Interactive color editing of nerfs
via decomposition-aware weight optimization,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), Oc-
tober 2023, pp. 3491–3501.
[15] A. Mazzucchelli, A. Garcia-Garcia, E. Garces, F. Rivas-Manzaneque,
F. Moreno-Noguer, and A. Penate-Sanchez, “Irene: Instant recoloring of
neural radiance fields,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), June 2024.
[16] B.
Wang,
N.
S.
Dutt,
and
N.
J.
Mitra,
“Proteusnerf:
Fast
lightweight nerf editing using 3d-aware image context,” arXiv preprint
arXiv:2310.09965, 2023.
[17] M. Gross and H. Pfister, Point-based graphics.
Elsevier, 2011.

<!-- page 14 -->
14
[18] M. Zwicker, M. Pauly, O. Knoll, and M. Gross, “Pointshop 3d: An
interactive system for point-based surface editing,” ACM Transactions
on Graphics (TOG), vol. 21, no. 3, pp. 322–329, 2002.
[19] Y. Ohtake, A. Belyaev, M. Alexa, G. Turk, and H.-P. Seidel, “Multi-level
partition of unity implicits,” in Acm Siggraph 2005 Courses, 2005, pp.
173–es.
[20] S. M. Seitz, B. Curless, J. Diebel, D. Scharstein, and R. Szeliski,
“A comparison and evaluation of multi-view stereo reconstruction al-
gorithms,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), vol. 1.
IEEE, 2006, pp. 519–
528.
[21] T. Thonat, E. Shechtman, S. Paris, and G. Drettakis, “Multi-view
inpainting for image-based scene editing and rendering,” in 2016 Fourth
International Conference on 3D Vision (3DV).
IEEE, 2016, pp. 351–
359.
[22] S.-H. Baek, I. Choi, and M. H. Kim, “Multiview image completion
with space structure propagation,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2016,
pp. 488–496.
[23] N. Snavely, S. M. Seitz, and R. Szeliski, “Photo tourism: exploring photo
collections in 3d,” in ACM siggraph 2006 papers, 2006, pp. 835–846.
[24] G. Chaurasia, S. Duchene, O. Sorkine-Hornung, and G. Drettakis,
“Depth synthesis and local warps for plausible image-based navigation,”
ACM transactions on graphics (TOG), vol. 32, no. 3, pp. 1–12, 2013.
[25] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[26] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian,
and W. Xinggang, “4d gaussian splatting for real-time dynamic scene
rendering,” in The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2024.
[27] D. Das, C. Wewer, R. Yunus, E. Ilg, and J. E. Lenssen, “Neural
parametric gaussians for monocular non-rigid object reconstruction,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[28] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow: 4d reconstruction
with dynamic 3d gaussian particle,” in The IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2024.
[29] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting
for real-time dynamic view synthesis,” in The IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2024.
[30] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in The IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2024.
[31] Z. Li, Z. Zheng, L. Wang, and Y. Liu, “Animatable gaussians: Learning
pose-dependent gaussian maps for high-fidelity human avatar modeling,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024.
[32] M. Kocabas, R. Chang, J. Gabriel, O. Tuzel, and A. Ranjan, “Hugs:
Human gaussian splats,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2024.
[33] R. Abdal, W. Yifan, Z. Shi, Y. Xu, R. Po, Z. Kuang, Q. Chen, D.-Y.
Yeung, and G. Wetzstein, “Gaussian shell maps for efficient 3d human
generation,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024.
[34] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu,
“Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-
time human novel view synthesis,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[35] L. Hu, H. Zhang, Y. Zhang, B. Zhou, B. Liu, S. Zhang, and L. Nie,
“Gaussianavatar: Towards realistic human avatar modeling from a sin-
gle video via animatable 3d gaussians,” in IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024.
[36] J. Xiang, X. Gao, Y. Guo, and J. Zhang, “Flashavatar: High-fidelity head
avatar with efficient gaussian embedding,” in The IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2024.
[37] H. Pang, H. Zhu, A. Kortylewski, C. Theobalt, and M. Habermann, “Ash:
Animatable gaussian splats for efficient and photoreal human rendering,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[38] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, and S. Tang, “3dgs-avatar:
Animatable avatars via deformable 3d gaussian splatting,” in The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[39] Y. Yuan, X. Li, Y. Huang, S. De Mello, K. Nagano, J. Kautz, and
U. Iqbal, “Gavatar: Animatable 3d gaussian avatars with implicit mesh
learning,” in The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2024.
[40] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and
G. Drettakis, “A hierarchical 3d gaussian representation for real-time
rendering of very large datasets,” ACM Transactions on Graphics,
vol. 44, no. 3, 2024.
[41] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
2023.
[42] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash,
“Compact3d: Compressing gaussian splat radiance field models with
vector quantization,” arXiv preprint arXiv:2311.18159, 2023.
[43] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d
scene representation via self-organizing gaussian grids,” 2023.
[44] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2024.
[45] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June
2024.
[46] D. Verbin, P. Hedman, B. Mildenhall, T. Zickler, J. T. Barron, and P. P.
Srinivasan, “Ref-nerf: Structured view-dependent appearance for neural
radiance fields,” in 2022 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
IEEE, 2022, pp. 5481–5490.
[47] T. Wu, J.-M. Sun, Y.-K. Lai, and L. Gao, “De-nerf: Decoupled neural ra-
diance fields for view-consistent appearance editing and high-frequency
environmental relighting,” in ACM SIGGRAPH 2023 conference pro-
ceedings, 2023, pp. 1–11.
[48] J. Zhang, Y. Yao, S. Li, J. Liu, T. Fang, D. McKinnon, Y. Tsin,
and L. Quan, “Neilf++: Inter-reflectable light fields for geometry and
material estimation,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2023, pp. 3601–3610.
[49] H. Wu, Z. Hu, L. Li, Y. Zhang, C. Fan, and X. Yu, “Nefii: Inverse render-
ing for reflectance decomposition with near-field indirect illumination,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 4295–4304.
[50] S. Wu, S. Basu, T. Broedermann, L. Van Gool, and C. Sakaridis, “Pbr-
nerf: Inverse rendering with physics-based neural fields,” arXiv preprint
arXiv:2412.09680, 2024.
[51] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “Gs-ir: 3d gaussian
splatting for inverse rendering,” in Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024, pp. 21 644–
21 653.
[52] S. Lai, L. Huang, J. Guo, K. Cheng, B. Pan, X. Long, J. Lyu, C. Lv, and
Y. Guo, “Glossygs: Inverse rendering of glossy objects with 3d gaussian
splatting,” IEEE Transactions on Visualization and Computer Graphics,
2025.
[53] Y. Shi, Y. Wu, C. Wu, X. Liu, C. Zhao, H. Feng, J. Zhang, B. Zhou,
E. Ding, and J. Wang, “Gir: 3d gaussian inverse rendering for relightable
scene factorization,” arXiv preprint arXiv:2312.05133, 2023.
[54] K. Ye, Q. Hou, and K. Zhou, “3d gaussian splatting with deferred
reflection,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–
10.
[55] Y. Endo, S. Iizuka, Y. Kanamori, and J. Mitani, “Deepprop: Extracting
deep features from a single image for edit propagation,” in Computer
Graphics Forum, vol. 35, no. 2.
Wiley Online Library, 2016, pp. 189–
201.
[56] X. An and F. Pellacini, “Appprop: all-pairs appearance-space edit
propagation,” ACM Transactions on Graphics (TOG), vol. 27, no. 3,
pp. 1–9, 2008.
[57] X. Chen, D. Zou, Q. Zhao, and P. Tan, “Manifold preserving edit
propagation,” ACM Transactions on Graphics (TOG), vol. 31, no. 6,
pp. 1–7, 2012.
[58] “Scribbleboost: Adding classification to edge-aware interpolation of
local image and video adjustments,” in Computer Graphics Forum,
vol. 27, no. 4.
Wiley Online Library, 2008, pp. 1255–1264.
[59] S. Meyer, V. Cornill`ere, A. Djelouah, C. Schroers, and M. Gross, “Deep
video color propagation,” in Proceedings of the British Machine Vision
Conference (BMVC), Newcastle upon Tyne, UK, September 3-6, 2018.
British Machine Vision Association (BMVA), 2018.
[60] B. Zhang, M. He, J. Liao, P. V. Sander, L. Yuan, A. Bermak, and
D. Chen, “Deep exemplar-based video colorization,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2019, pp. 8052–8061.

<!-- page 15 -->
15
[61] H. Chang, O. Fried, Y. Liu, S. DiVerdi, and A. Finkelstein, “Palette-
based photo recoloring,” ACM Transactions on Graphics (TOG), vol. 34,
no. 4, pp. 1–11, 2015.
[62] J. Tan, J. Echevarria, and Y. Gingold, “Efficient palette-based decom-
position and recoloring of images via rgbxy-space geometry,” ACM
Transactions on Graphics (TOG), vol. 37, no. 6, pp. 1–10, 2018.
[63] Z.-J. Du, K.-X. Lei, K. Xu, J. Tan, and Y. Gingold, “Video recoloring
via spatial-temporal geometric palettes,” ACM Transactions on Graphics
(TOG), vol. 40, no. 4, Aug. 2021.
[64] Z. Chen, F. Wang, and H. Liu, “Text-to-3d using gaussian splatting,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[65] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Luciddreamer:
Towards high-fidelity text-to-3d generation via interval score matching,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[66] H. Ling, S. W. Kim, A. Torralba, S. Fidler, and K. Kreis, “Align
your gaussians: Text-to-4d with dynamic 3d gaussians and composed
diffusion models,” in The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2024.
[67] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, “Sc-
gs: Sparse-controlled gaussian splatting for editable dynamic scenes,”
in The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[68] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai,
L. Yang, H. Liu, and G. Lin, “Gaussianeditor: Swift and controllable
3d editing with gaussian splatting,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[69] H. Yu, J. Julin, Z. A. Milacski, K. Niinuma, and L. A. Jeni, “Cogs:
Controllable gaussian splatting,” in Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR), 2024.
[70] M. Zwicker, H. Pfister, J. Baar, and M. Gross, “Surface splatting,” Pro-
ceedings of the ACM SIGGRAPH Conference on Computer Graphics,
vol. 2001, 08 2001.
[71] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), June 2022, pp. 5470–5479.
[72] S. A. Shafer, “Using color to separate reflection components,” Color
Research & Application, vol. 10, no. 4, pp. 210–218, 1985.
[73] S. Beigpour and J. Van De Weijer, “Object recoloring based on intrinsic
image estimation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR).
IEEE, 2011, pp.
327–334.
[74] M. Pharr, W. Jakob, and G. Humphreys, Physically based rendering:
From theory to implementation.
MIT Press, 2023.
[75] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ra-
mamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view
synthesis with prescriptive sampling guidelines,” ACM Transactions on
Graphics (TOG), 2019.
[76] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and
R. B. Girshick, “Segment anything,” 2023 IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 3992–4003, 2023. [Online].
Available: https://api.semanticscholar.org/CorpusID:257952310
[77] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to
follow image editing instructions,” in CVPR, 2023.
