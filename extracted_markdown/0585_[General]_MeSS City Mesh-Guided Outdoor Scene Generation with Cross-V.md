<!-- page 1 -->
MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent
Diffusion
Xuyang Chen1,3, Zhijun Zhai2, Kaixuan Zhou3*, Zengmao Wang2*, Jianan He3, Dong Wang3,
Yanfeng Zhang3, Mingwei Sun2,3, R¨udiger Westermann1, Konrad Schindler4, Liqiu Meng1
1TU Munich, 2Wuhan University, 3Huawei Riemann Lab, 4ETH
MeSS
Stylized 
Renderer
City Surface Model
Gaussian Field
Stylized Videos
Figure 1: Starting from textureless urban meshes, our MeSS synthesizes high-quality Gaussian Splatting Scenes with realistic
appearance. After synthesis, these Gaussian scenes can be further rendered into stylized videos.
Abstract
Mesh models have become increasingly accessible for nu-
merous cities; however, the lack of realistic textures restricts
their application in virtual urban navigation and autonomous
driving. To address this, this paper proposes MeSS (Mesh-
based Scene Synthesis) for generating high-quality, style-
consistent outdoor scenes with city mesh models serving as
the geometric prior. While image and video diffusion mod-
els can leverage spatial layouts (such as depth maps or HD
maps) as control conditions to generate street-level perspec-
tive views, they are not directly applicable to 3D scene gen-
eration. Video diffusion models excel at synthesizing consis-
tent view sequences that depict scenes but often struggle to
adhere to predefined camera paths or align accurately with
rendered control videos. In contrast, image diffusion mod-
els, though unable to guarantee cross-view visual consistency,
can produce more geometry-aligned results when combined
with ControlNet. Building on this insight, our approach en-
hances image diffusion models by improving cross-view con-
sistency. The pipeline comprises three key stages: first, we
generate geometrically consistent sparse views using Cas-
caded Outpainting ControlNets; second, we propagate denser
intermediate views via a component dubbed AGInpaint; and
third, we globally eliminate visual inconsistencies (e.g., vary-
ing exposure) using the GCAlign module. Concurrently with
generation, a 3D Gaussian Splatting (3DGS) scene is recon-
structed by initializing Gaussian balls on the mesh surface.
Our method outperforms existing approaches in both geo-
metric alignment and generation quality. Once synthesized,
the scene can be rendered in diverse styles through relighting
*project lead
and style transfer techniques.
1
Introduction
Projects like 3DCityDB (Yao, Chaturvedi, and Kolbe 2025),
Hamburger LOD3 (Landesbetrieb Geoinformation und Ver-
messung Hamburg 2025), Ingolstadt LOD3 (SAVeNoW
2020), and 3DBAG (Peters et al. 2022) provide detailed city-
scale mesh models, typically at Level of Detail 3 (LoD3)
for buildings. However, these models often lack high quality
textures, relying instead on simple templates—or omitting
textures entirely. This limitation hinders many downstream
applications. For example, in VR-based urban navigation,
texture quality is essential to user experience—especially at
street level, where users are close enough to notice visual
flaws.
While deploying data collection vehicles to gather addi-
tional street-view imagery may seem like a direct solution,
it is prohibitively costly and still prone to issues such as oc-
clusions and illumination variability. A more scalable and
customizable alternative is to leverage generative models to
enrich the visual appearance of mesh models. This approach
is both cost-efficient and adaptable, enabling users to tailor
scene aesthetics to their specific needs or preferences.
Recent works like (Chai et al. 2023; Lin et al. 2023; Chen,
Wang, and Liu 2023; Xie et al. 2024a,b, 2025) have investi-
gated to generate a city as NeRF (Mildenhall et al. 2021) or
3DGS (Kerbl et al. 2023) representations from city semantic
& height map. Since their training process is supervised by
images in birds-eye-view, their visual quality degrades sig-
arXiv:2508.15169v3  [cs.CV]  4 Jan 2026

<!-- page 2 -->
ControlNet-S
Multi-
Control 
Signal
ControlNet-N
Outpainting
̃𝐼𝐼𝑁𝑁
Stage II: Gaussian Field Densification  (Sec 3.3)
Multi-
Control 
Signal
Multi-
Control 
Signal
ControlNet-N 
Outpainting
…
…
…
…
2D Gaussians 
Registration
Stage I: Sparse Gaussian Field Construction (Sec 3.2)
2D Gaussians 
Registration
𝐼𝐼𝑁𝑁
𝐼𝐼𝑁𝑁−1
𝐼𝐼0
Key Frames Generation
Appearance Guided Inpainting
Denoising
Guidance
𝑪𝑪𝑵𝑵
𝑪𝑪𝑵𝑵−𝟏𝟏
…
Depth
Semantics
Normal
Gaussian Field
update
𝑪𝑪n
𝑪𝑪n+𝟏𝟏
Render
Global Consistency Alignment (Sec 3.4)
Multi-view 
Denoising
Optimize
Render 
& Add 
noise
Key 
Frame
Views
Interm.  
Frame 
Views
2D Gaussians
…
…
Figure 2: Schematic illustration of MeSS. Given a sequence of camera poses, we start by generating the last viewpoint
using a ControlNet-s. Then we generate other key views in reverse order using a ControlNet-n, while transferring information
backwards through the sequence. All generated pixels are projected onto the mesh surface as 2D Gaussian surfels. From
the resulting Gaussian field, intermediate views are rendered and filled up with Appearance-Guided Inpainting (AGInpaint),
simultaneously densifying the Gaussian field. Each time the field is extended, a Global Consistency Alignment ensures spatial
consistency by simultaneously denoising multi-view renderings.
nificantly after zooming-in. This in turn has fueled efforts
to synthetically generate visual data from the perspective on
the street.
To alleviate the cost and complexity of real-world data
collection for autonomous driving, recent approaches (Wang
et al. 2024b; Li, Zhang, and Ye 2024; Zhao et al. 2024a;
Gao et al. 2025) have explored generating driving scenes di-
rectly from video diffusion models (Blattmann et al. 2023).
These methods primarily focus on synthesizing foreground
elements such as traffic participants and road infrastructure,
leaving the background—particularly buildings—to be hal-
lucinated by the generative model without explicit structural
constraints. As a result, the generated videos often fail to
align with the actual city layout or urban topology.
To tackle this limitation, Streetscapes (Deng et al. 2024)
leverages simplified building geometries and map layouts
as conditioning inputs to enhance geometric consistency.
Similarly, Cosmos-Transfer1 (Alhaija et al. 2025) integrates
background structures into the generation process by utiliz-
ing sparse depth data derived from lidar point clouds, though
its depth control mechanism ignores all scene content be-
yond a 75-meter range. Nevertheless, due to the imprecise
nature of their geometric inputs and the feedforward design
of their conditional generation pipelines, these methods fail
to guarantee precise geometric alignment.
Besides, video diffusion-based methods frequently suffer
from temporal artifacts, manifesting as abrupt transitions or
inconsistencies across frames. These limitations collectively
highlight the necessity of constraining generation to adhere
to actual mesh surfaces.
Directly texturing mesh models with 2D diffusion mod-
els (Chen et al. 2023a; Metzer et al. 2023; Chen et al. 2024;
Richardson et al. 2023) may seem a viable option, but ex-
isting methods are limited to single objects or small scenes
with low polygon counts, far from city-scale street scenes.
Another approach is perpetual view generation from a sin-
gle image (Yu et al. 2024a,b; Wang et al. 2024a; Chung et al.
2023), which relies on progressively outpainting of RGB im-
ages and associated depth maps to cover the scene, using
2D diffusion models (Rombach et al. 2022). Their outpaint-
ing strategy often gradually drifts away from the initial ap-
pearance and is not capable for generating consistent dense
views.
In summary, one must solve two main challenges to en-
able the scene generation with generative models: (i) Gener-
ate visual results aligning with given city surface models. (ii)
Maintain a consistent appearance across long range without
drifting.
We carefully design the MeSS pipeline to meet these re-
quirements. A common approach for generating consecu-
tive view sequences with 2D diffusion models is the warp-
and-outpaint procedure, we optimize it by a two-staged
framework. In Stage I, Sparse key frames are generated
by two Cascaded Outpainting ControlNets, which consec-
utively outpaint new frames from preceding ones. With it,
appearance drifts among key frames can be largely reduced
in a long range. Meanwhile, All key frames are spread as
Gaussian balls on the surface of mesh model to guarantee
the geometry alignment. Sparse-view reconstructed Gaus-
sian Scene commonly depicts blurry artifacts or silhoutte in
denser novel views. In Stage II, we fix these issues by opti-
mizing the Gaussian Scene at these views.
To that end, we make the following technical contribu-
tions:

<!-- page 3 -->
• We introduce Cascaded Outpainting ControlNets con-
ditioned on preceding frame to generate consistent key
frames to ensure long range consistency.
• We devise a Appearance Guided Inpainting method to
meticulously inpaint the occluded area in the intermedi-
ate dense views using the guidance of surrounding known
regions.
• Our method achieves state-of-the-art (SOTA) perfor-
mance in generating 3D Scene from city mesh models,
ensuring high appearance fidelity and cross-view consis-
tency.
2
Related Works
2.1
Scene Generation from 2D Map
InfiniCity (Lin et al. 2023) decomposes 3D scene genera-
tion into three steps: 2D map generation, map-to-voxel lift-
ing, and voxel texturing through neural rendering. Scene-
Dreamer (Chen, Wang, and Liu 2023) represents unbounded
3D landscapes via a bird’s-eye view (BEV) layout with
height and semantic fields, manipulated through 2D genera-
tive models. Recent extensions like CityDreamer (Xie et al.
2024a) and its temporal variant CityDreamer4D (Xie et al.
2025) refine this paradigm by decomposing the rendering
process into specialized modules for distinct scene com-
ponents. CityDreamer4D explicitly separates static back-
grounds, dynamic buildings, and movable foreground ob-
jects, while incorporating temporal consistency for 4D ur-
ban scene synthesis. Despite these advancements, existing
methods (Xie et al. 2024b,a, 2025) still face challenges in
stabilizing layout generation and achieving high-fidelity ren-
dering.
2.2
Perpetual View Generation
Infinite-Nature (Liu et al. 2021) defines perpetual view gen-
eration as synthesizing views along arbitrary camera trajec-
tories from a single image, employing a render-and-refine
pipeline that progressively extends scenes via outpainting
with cross-frame depth alignment. This principle has been
extended through GANs (Li et al. 2022) and diffusion mod-
els: DiffDreamer (Cai et al. 2023) and SceneScape (Frid-
man et al. 2023) focus on static scene extrapolation, while
WonderWorld (Yu et al. 2024a) enables interactive 3D scene
generation through guided depth diffusion, allowing users to
specify scene contents and viewpoints. For view-consistent
generation, VistaDream (Wang et al. 2024a) explicitly en-
forces multiview geometry constraints during denoising pro-
cess. These and other similar approaches (Chung et al. 2023;
Yu et al. 2024b) all profit from advances in monocular depth
estimation (Ranftl et al. 2020; Bhat et al. 2023; Ke et al.
2024a). However, they still exhibit rapid appearance drift in
complex scenarios like urban driving, where error accumula-
tion in the outpainting step amplifies inconsistencies across
occluded regions.
2.3
Scene Synthesis from Video Generation
In the context of scene construction, video generators of-
fer a way to synthesize (approximately) consistent views,
either by conditioning on a single frame (Yu et al. 2024c;
Fan et al. 2024; Sun et al. 2024) or by interpolating multi-
ple given viewpoints (Liu et al. 2024; Yu et al. 2024c; Sun
et al. 2024). In particular, StreetScapes (Deng et al. 2024)
addresses a task similar to ours with a two-frame video gen-
erator based on AnimateDiff (Guo et al. 2023a). Rendered
depth, height, and semantics are used as control inputs to
generate consecutive street views. Despite starting from a
pretrained Latent Diffusion Model (Rombach et al. 2022),
training such a video generator is expensive in terms of train-
ing data and compute. The insufficient geometric controling
ability also leads to adjacent-frame inconsistency and tem-
poral drift in long sequence.
3
Methods
The MeSS pipeline (Fig. 2) is designed to synthetically gen-
erate viewpoints to reconstruct gaussian scene following a
sparse-to-dense scheme. Given a 3D city map (i.e., a mesh
model with semantic and instance labels but without tex-
ture), we specify a virtual camera path via a sequence of M
views.
In Stage I, we generate a subset of N key view im-
ages along the sequence via a warp-and-outpaint procedure:
starting from the initial key frame generated by geometric-
conditioned ControlNet-s, each proceeding key frame is
warped as an additional condition to the outpainting of new
key frame with ControlNet-n (Sec. 3.2). After obtaining
all key frames, we use them to construct a Gaussian field
through optimizing gaussian surfels on the surface of mesh
models.
In Stage II, we render from gaussian scene the interme-
diate views {Cl}, l ∈

1,
M
N−1 −2

between each pair of
subsequent key views. Artifacts like silhouettes in interme-
diate frames are filled up by Appearance Guided Inpainting
(Sec. 3.3). Lastly, Global Consistency Alignment (Sec. 3.4)
further enhances the appearance consistency of gaussian sur-
fels learned from different views.
3.1
Preliminaries
Latent Consistency Model.
We utilize a Latent Consis-
tency Model (LCM) to enhance the inpainting consistency
across views in our method. Its core idea is to learn a func-
tion that maps any point on a trajectory of the probability
flow ODE (Song et al. 2020; Lu et al. 2022) to that trajec-
tory’s origin (i.e., the solution of the ODE). LCM is trained
by enforcing the self-consistency property with a consis-
tency function fθ:
f θ (zt, c, t) = f θ (zt′, c, t′) , ∀t, t′ ∈[δ, T],
(1)
with δ a fixed, small positive number and T refers total diffu-
sion steps. For ϵ-prediction (Song et al. 2020), the function
f θ is parameterized as
f θ(zt, c, t) = cskip(t)zt + cout(t) (zt−σ(t)ˆϵθ(zt,c,t))
α(t)
,
(2)
where cskip(t) and cout(t) are scaling factors dependent on
timestep t, and ˆϵθ(zt, c, t) is the noise prediction model of
LCM. During inference, the noise-free estimate ˜z0 can be
found as ˜z0 = f θ(zt, c, t) with a single denoising step.

<!-- page 4 -->
3.2
Stage I: Gaussian Field Construction from
Key Views
Key Views Generation with Cascaded ControlNets.
Two multi-conditioned ControlNets (Zhang, Rao, and
Agrawala 2023) are introduced to generate N key views in
autoregressive fashion, moving backwards from the end to
the start of the camera path*. First, ControlNet-S takes con-
trol signals {dN, sN, nN} (depth, semantics and normals)
at camera pose CN to generate view image IN. As single-
channel disparity maps may lose the ability to discriminate
depth values in the far field, so we encode depth as a col-
ormap. This ensures depth variations are retained both near
the camera and in the far field.
Having generated the last view image IN, we warp the
image content to its previous key view CN−1 via the known
depth map. The reason for working backwards from the
last frame is that, in a forward-facing camera, the warp-
ing will contract the pixel coordinates towards the image
center. In this way, image generation artifacts due to lim-
itations of the ControlNet, which mostly occur in the far
field, will be diminished, whereas forward warping would
amplify them and cause error build-up. To fill the peripheral
regions where warped content is not available, ControlNet-n
outpaints the missing values. On top of the inputs also used
by ControlNet-s it takes the reference image ˜I
N−1 as ad-
ditional condition to generate the final IN−1(See Appendix
for more details). The alternation between warping and out-
painting is repeated until the first view of the sequence is
reached.
However, the extended key frames generated with the de-
scribed warp-and-outpaint scheme may exhibit noticeable
seams (Fig. 4b left) due to different exposures. To remove
them and achieve coherent appearance, we synchronize all
key frames with a Global Consistency Alignment, which will
be discussed in Sec. 3.4).
Gaussian Field Construction On Mesh Surface
From
the set of keyframes {In}, n ∈(0, N), a 3D Gaussian field
is constructed. For the last frame IN (i.e. the first generated
image) we instantiate a Gaussian splat for every pixel. In
subsequent key frames In, further Gaussians are added to
fill in regions with low opacity values in the Gaussian field,
in other words we also ”outpaint” the Gaussians.
To avoid the computational cost associated with Gaus-
sian Field optimization, splats are directly positioned on
the known 3D surface by lifting image pixels according
to the depth map. Furthermore, following the approach
in SUGAR(Gu´edon and Lepetit 2024), the Gaussians are
aligned with the surface and flattened by reducing the scale
factor along the surface normal to a small value. The other
two scale parameters are selected to ensure full coverage
of the mesh surface, thereby preventing the formation of
Moir´e patterns and gaps (Yu et al. 2024a). For the Gaus-
sians, Spherical Harmonics coefficients are computed from
the RGB values of corresponding pixels, while opacity is set
based on empirical observations. Additional details are pro-
vided in the supplementary material.
*Views are indexed in driving direction.
3.3
Stage II: Gaussian Field Densification
Appearance Guided Inpainting
We render from above
constructed scene at intermediate novel views {Cl} between
two keyframes In and In+1, n ∈[1, N). As illustrated in
Fig. 3, the novel views are inspected with silhouettes since
they are rendered from the coarse scene model. Novel views
not included during the 3DGS reconstruction commonly suf-
fer from holes and blurry textures (Kerbl et al. 2023). In-
spired by (Epstein et al. 2023; Dhariwal and Nichol 2021;
Luo et al. 2024; Yu et al. 2024a), we modify the sampling
process of latent diffusion to obtain a training-free method
for RGB inpainting, termed Appearance Guided Inpainting
(AGInpaint). Specifically, AGInpaint rectifies the prediction
Figure 3:
Silhouettes
on
novel views, marked with
red ellipses and arrows.
of the score function ϵθ to the direction, that minimizes the
discrepancy between diffusion prediction ˜x0 † and known
RGB value x0 in the unmasked region M. In simple words,
the diffusion process should predict correct RGB values for
known regions. This objective guides the diffusion model
to predict aligned inpainting with neighbor pixels. At each
sampling step, the rectification process is repeated Ng times,
incrementally updating the latent with a low learning rate
lr. This corresponds to a gradient descent towarda a local
optimum, thus preserving sharper details and more consis-
tent texture within the reference image. The guided rectifi-
cation is beneficial already in the initial stages of the sam-
pling process: at earlier timesteps it aligns the overall color
palette and dominant hues with those of the reference im-
age, at later timesteps, the progressive refinement corrects
fine-grained details and textures. For optimal performance
of the guided inapinting we adopt LCM. Its self-consistency
property Eq. (1) enables consistent estimation of the orig-
inal sample x0 and stabilizes the guidance across different
timesteps during diffusion sampling.
The complete process for guided inpainting is spelled out
in Algorithm 1, where ˜x0 is the noise-free one-step estimate
of xt from the Consistency Function Eq. (2). M denotes the
unmasked region. To adaptively regulate the step size, we
scale the gradients proportional to the mean absolute magni-
tude of the latent, respectively the gradients; thus accelerat-
ing convergence. The scaling is combined with the learning
rate lr and denoted as st.
All newly inpainted pixels are spread onto the mesh sur-
face in the aforementioned way.
3.4
Global Consistency Alignment
Gaussian field generated with the above two stages so far
still exhibit appearance artifacts such as brightness drift, out-
painting mistakes, etc. To resolve those, we draw inspira-
tion from Multiview Consistency Sampling (MCS) in Vis-
†By a slight abuse of notation, we skip the step from pixel space
to latent space for simplicity.

<!-- page 5 -->
Algorithm 1: Appearance Guided Sampling
1: xT ∼N(0, I)
2: for t = T, . . . , 1 do
3:
ˆϵ1
t ←ϵθ (xt, c, t)
4:
for ng = 1, . . . , Ng do
5:
gt = ∇dtSmoothL1(˜x0 ⊙M, x0 ⊙M)
6:
st = lr ×
mean(|ˆϵ
ng
t
|)
mean(|gt|)
7:
ˆϵ
ng+1
t
= ˆϵ
ng
t
+ stgt
8:
end for
9:
xt−1 =
1
√αt

xt −
1−αt
√1−¯αt ˆϵ
Ng
t

10: end for
11: return x0
taDream (Wang et al. 2024a) and design a cleaning method
called Global Consistency Alignment (GCA), tailored for
our sparse-to-dense generation pipeline.
After the gaussian field construction process detailed in
Sec. 3.2, we rerender a sequence of images x(1:N) at the
same camera viewpoints C(1:N). We then apply the forward
diffusion process to obtain x(1:N)
T
by adding T1 steps of
noise to x(1:N). Using a learned LCM, we then derive a
batch of noise-free estimates ˜x(1:N)
0
through Eq. (2) and ad-
just it to ¯x(1:N)
t
via
¯x(n)
t
= wtγ(n)
t
x′(n)
t
+ (1 −wt)˜x(n)
0 , n ∈(1, N)
(3)
where γ(n)
t
= std(˜x(n)
0 )/std(x(n)
t
) balances the exposure
and wt is a weight that governs the trade-off between the
rendered x′(n)
t
and the ˜x(n)
0
estimated by multi-view consis-
tent denoising.
Next, the adjusted estimates ¯x(1:N)
t
serve to refine the
Gaussian field, while x′(1:N)
t
are rendered from the updated
scene after each refinement step. Here, ¯x(1:N)
t
achieves a
balance between x′(1:N)
t
and ˜x(1:N)
0
, thus ensuring multi-
view consistency while at the same time enhancing details
through reverse diffusion.
Finally, the same method is used to also improve the con-
sistency of views C(n:(n+1)k), i.e., the sub-sequence of in-
termediate frames between (and including) two consecutive
keyframes.
4
Experiments
4.1
Data Preparation and Implementation
To assess the feasibility of our pipeline, we render RGB im-
ages with corresponding depth, normal, and semantic maps
using City Sample Project (Epic Games 2022) in Unreal En-
gine 5 (Epic Games 2021). ControlNets are trained upon
frozen Stable Diffusion 1.5 (SD1.5) with training data ren-
dered above. More details about data preparation and imple-
mentation can be found in the supplementary material.
Our pipeline is designed to address the challenge of scene
generation on a predefined city mesh layout using image dif-
fusion model. We compare our approach with other methods
in the domains of 3D city generation, scene generation based
on video synthesis, and perpetual view generation. For quan-
titative evaluation of the generated results, we employ the
Fr´echet Inception Distance (FID) and Kernel Inception Dis-
tance (KID) metrics. These metrics measure the discrepancy
between the distributions of the generated results and the
ground truth imagery, providing a standard benchmark for
assessing the performance of generative models. To evaluate
the consistency across views, we calculate the Learned Per-
ceptual Image Patch Similarity (LPIPS) for extended views
generated from a real image, offering insights into the per-
ceptual quality and coherence of the synthesized sequences
(a)
(b)
Figure 4: a) The comparison result of Resample(left) vs.
AGInpaint(right). AGInpaint performs better than Resam-
ple in slim region inpainting b) The comparison of results
w/o(left) and w/(right) GCAlign. GCAlign is able to harmo-
nize the seams brought by different exposures
4.2
Comparison with other methods
Baselines.
Both WonderWorld (Yu et al. 2024a) and Vis-
taDream (Wang et al. 2024a) rely on monocular depth esti-
mation methods (Bochkovskii et al. 2024; Ke et al. 2024b)
to give depth value for placing Gaussians into 3D space,
while our method is based on mesh geometry with pre-
cise metric depth. To form a fair comparison, we inject
metric depth into their pipelines and align their inpaint-
ing diffusion model with ours. The altered ones are named
as WonderWorld† and VistaDream† respectively. For each
camera path, We start the scene from the same initial view
and outpaint them by moving the camera backwards for 100
meters using the two and our pipeline. In this way, 3200 im-
ages are produced in total.
As depicted in Tab. 1, our method achieves better metrics.
We visualize several intermediate frames in Fig. 5 and ob-
serve that for the same outpainting task, we produce more
consistent results aligning well with the underlining geome-
try, while they suffer a lot from quality degeneration caused
by discrepancy accumulation of inpainting results. Even
equipped with Multiview Consistency Sampling(Sec3.2 in
Vistadream), Vistadream† does not show apparent improve-
ment wrt. WonderWorld†, which emphasizes the synchroni-
sation of Multiview Consistency Alignment with consistent
inpainting, as ours.
Comparison with Perpetual View Generation
We also
visualize the result of the naive Wonderworld in Fig. 5.
Without any prior information about the building geometry
it cannot extend their facades and the outpainted content is
tending to be more irrelevant to the initial view. This verifies
the difficulties of their method on this task.

<!-- page 6 -->
CityDreamer4D
StreetScapes
VistaDream
WonderWorld
WonderWorld
MeSS(Ours)
Figure 5: Visual comparison with other methods. Since there is no code provided by CityDreamer4D and Streetscapes, we take
the visual results from their papers. Please zoom in to check for details.
Comparison
with
3D
City
Generation
City-
Dreamer4D
(Xie et al. 2025) is implemented on the
Citytopia dataset
(Xie et al. 2025), which is also con-
structed based on the City Sample Project in Unreal Engine
5 (UE5), similar to ours. In comparison, our method demon-
strates superior performance, as evidenced by lower scores
in the FID and KID metrics. Furthermore, as illustrated
in Fig. 5, their generated scenes exhibit notable limitations
in visual quality upon closer inspection, particularly at street
level. Specifically, the results lack critical urban elements
such as streetlamps, traffic signs, and other street-level in-
stances, significantly diminishing their realism and practical
applicability.
Comparison
with
Video-Based
Generation
For
StreetScapes
(Deng et al. 2024), we directly use the
results reported in their paper since their code is not
publicly available. However, forming a fair comparison
is challenging, as their model is trained on a significantly
larger dataset compared to ours‡. Given that our generated
sequence is longer than theirs (200 frames vs. 64 frames),
we evaluate their results within the range of 32 to 64 frames.
In terms of quality, our results are on par with theirs, while
we outperform them in terms of temporal consistency.
This advantage can be attributed to our method’s effective
utilization of geometric priors, which plays a key role in
achieving superior consistency.
‡Even for their experiment on the London patch, they utilize
approximately 20 times more data than we do.

<!-- page 7 -->
Original Views
Cosmos-Transfer1
"A realistic urban scene in sunny day "
TC-Light
Figure 6: Stylized rendering results with Cosmos-Transfer1 and TC-Light. Please zoom in to check for details.
4.3
Stylized rendering
As you can see, the appearance of our generated scene is
heavily branded with City Sample style, simply tweaking
text prompt during view generation does not work for styl-
ization. This hinders the application of the scene, so we
develop a way to customize it through stylized rendering.
Given a camera path in the scene, we render video and
transfer it to different style via video relightning method
TC-Light (Liu et al. 2025) or SDEdit (Meng et al. 2021)
with Cosmos-Transfer1 (Alhaija et al. 2025). As depicted in
Fig. 6, both are capable of completing this task by taking
the original views as prior, and Cosmos-Transfer1 outper-
forms TC-Light in visual fidelity. The styled videos can be
projected back to the gaussian scene if desired.
LPIPS↓
FID↓
KID↓
Citydreamer4D (Wang et al. 2024a)
-
88.48
0.049
WonderWorld-Geometry (Wang et al. 2024a)
0.516
75.807
0.076
Vistadream-Geometry (Yu et al. 2024a)
0.508
72.44
0.073
StreetScapes (Deng et al. 2024)
0.519
29.93
0.025
MeSS(Ours)
0.348
28.17
0.0161
Table 1: Quantitative evaluations on generated sequences
4.4
Ablation Study
Cascaded ControlNets
For the outpainting task of key
views in Stage I, we replace the preceding frame condi-
tioned ControlNet-N by ControlNet-S. To make it differ-
ent w.r.t. common warp-and-outpaint pipeline
(Yu et al.
2024a,b), AGInpaint is still equipped. As depicted in Tab. 2,
the fidelity of generated frames degenerates dramatically.
Besides, the consistency is also getting worse due to larger
appearance drift accumulated during scene extrapolation.
Appearance Guided Inpainting
We replace the AGIn-
paint component in our method with a simpler approach
called Resample (Deng et al. 2024; Lugmayr et al. 2022),
which involves re-adding several rounds of noise at each de-
noising step to achieve a more homogeneous inpainted re-
sult. As demonstrated in Tab. 2, the absence of AGInpaint
significantly impacts the generation quality. Through a vi-
sual comparison of the inpainted regions produced by AG-
Inpaint and Resample (Fig. 4a), we highlight the superior
efficacy of AGInpaint. Due to the downsampling of inpaint-
ing masks, Resample struggles to fill in slim regions with
Latent Diffusion Models, causing noticeable streak patterns.
ControlNet-n
AGInpaint
GCAlign
LPIPS
FID
KID
✗
0.382
54.54
0.0526
✗
0.422
51.12
0.0459
✗
0.346
26.25
0.0132
Table 2: Ablation studies on different components. ✗means
corresponding component is turned off.
Global Consistency Alignment
As shown in the top por-
tion of Fig. 4b, there is a noticeable lighting misalignment in
the generated scene when GCAlign is not applied. However,
after integrating GCAlign into our pipeline, the seams are
effectively harmonized, as demonstrated in the bottom por-
tion of Fig. 4b. An interesting observation is that GCAlign
has a trade-off: while it improves visual coherence, it tends
to introduce a slight blurring effect, leading to a loss of fine
details. This is also reflected in the lower FID/KID metrics.
Nevertheless, we consider this trade-off acceptable in favor
of achieving a more visually consistent appearance.
5
Conclusion
In this paper, we present a pipeline for generating gaus-
sian field from city surface models. Our method leverages
a meticulously designed outpainting procedure, which en-
sures strong alignment with predefined scene geometry. Ad-
ditionally, even after extensive view extrapolation, the newly
generated regions maintain a coherent appearance with the
preceding frames. This approach offers the advantage of low
training costs and paves the way for utilizing real-world city
3D maps to synthesize Gausisan scenes.
6
Future Works
In future, we expect to integrate more advanced base mod-
els such as FLUX.1 or SD3, then anticipate significant im-
provements in resolution and overall quality. Additionally,
we plan to explore the integration of geometric information
into video diffusion, enhancing applicability for scene gen-
eration.

<!-- page 8 -->
References
Alhaija, H. A.; Alvarez, J.; Bala, M.; Cai, T.; Cao, T.; Cha,
L.; Chen, J.; Chen, M.; Ferroni, F.; Fidler, S.; et al. 2025.
Cosmos-transfer1: Conditional world generation with adap-
tive multimodal control. arXiv preprint arXiv:2503.14492.
Bhat, S. F.; Birkl, R.; Wofk, D.; Wonka, P.; and M¨uller, M.
2023. Zoedepth: Zero-shot transfer by combining relative
and metric depth. arXiv preprint arXiv:2302.12288.
Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.;
Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.;
Letts, A.; et al. 2023. Stable video diffusion: Scaling la-
tent video diffusion models to large datasets. arXiv preprint
arXiv:2311.15127.
Bochkovskii, A.; Delaunoy, A.; Germain, H.; Santos, M.;
Zhou, Y.; Richter, S. R.; and Koltun, V. 2024. Depth pro:
Sharp monocular metric depth in less than a second. arXiv
preprint arXiv:2410.02073.
Cai, S.; Chan, E. R.; Peng, S.; Shahbazi, M.; Obukhov, A.;
Van Gool, L.; and Wetzstein, G. 2023.
Diffdreamer: To-
wards consistent unsupervised single-view scene extrapola-
tion with conditional diffusion models. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, 2139–2150.
Chai, L.; Tucker, R.; Li, Z.; Isola, P.; and Snavely, N.
2023. Persistent nature: A generative model of unbounded
3d worlds. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 20863–20874.
Chen, D. Z.; Li, H.; Lee, H.-Y.; Tulyakov, S.; and Nießner,
M. 2024. Scenetex: High-quality texture synthesis for in-
door scenes via diffusion priors.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 21081–21091.
Chen, D. Z.; Siddiqui, Y.; Lee, H.-Y.; Tulyakov, S.; and
Nießner, M. 2023a. Text2tex: Text-driven texture synthesis
via diffusion models. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, 18558–18568.
Chen, R.; Chen, Y.; Jiao, N.; and Jia, K. 2023b. Fantasia3d:
Disentangling geometry and appearance for high-quality
text-to-3d content creation. In Proceedings of the IEEE/CVF
international conference on computer vision, 22246–22256.
Chen, Z.; Wang, G.; and Liu, Z. 2023. Scenedreamer: Un-
bounded 3d scene generation from 2d image collections.
IEEE transactions on pattern analysis and machine intel-
ligence.
Chung, J.; Lee, S.; Nam, H.; Lee, J.; and Lee, K. M. 2023.
Luciddreamer: Domain-free generation of 3d gaussian splat-
ting scenes. arXiv preprint arXiv:2311.13384.
Deitke, M.; Schwenk, D.; Salvador, J.; Weihs, L.; Michel,
O.; VanderBilt, E.; Schmidt, L.; Ehsani, K.; Kembhavi, A.;
and Farhadi, A. 2023. Objaverse: A universe of annotated
3d objects. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 13142–13153.
Deng, B.; Tucker, R.; Li, Z.; Guibas, L.; Snavely, N.; and
Wetzstein, G. 2024.
Streetscapes: Large-scale consistent
street view generation using autoregressive video diffusion.
In ACM SIGGRAPH 2024 Conference Papers, 1–11.
Dhariwal, P.; and Nichol, A. 2021. Diffusion models beat
gans on image synthesis. Advances in neural information
processing systems, 34: 8780–8794.
Epic Games. 2021.
Unreal Engine 5.
https://www.
unrealengine.com/en-US/unreal-engine-5. Accessed: 2025-
08-01.
Epic Games. 2022.
City Sample Project.
https://www.
unrealengine.com/marketplace/en-US/product/city-sample.
Accessed: 2025-08-01.
Epstein, D.; Jabri, A.; Poole, B.; Efros, A.; and Holynski, A.
2023. Diffusion self-guidance for controllable image gener-
ation. Advances in Neural Information Processing Systems,
36: 16222–16239.
Fan, Z.; Wen, K.; Cong, W.; Wang, K.; Zhang, J.; Ding,
X.; Xu, D.; Ivanovic, B.; Pavone, M.; Pavlakos, G.; et al.
2024. InstantSplat: Sparse-view SfM-free Gaussian Splat-
ting in Seconds. arXiv preprint arXiv:2403.20309.
Fridman, R.; Abecasis, A.; Kasten, Y.; and Dekel, T.
2023. Scenescape: Text-driven consistent scene generation.
Advances in Neural Information Processing Systems, 36:
39897–39914.
Gao, S.; Yang, J.; Chen, L.; Chitta, K.; Qiu, Y.; Geiger, A.;
Zhang, J.; and Li, H. 2025.
Vista: A generalizable driv-
ing world model with high fidelity and versatile controlla-
bility. Advances in Neural Information Processing Systems,
37: 91560–91596.
Garland, M.; and Heckbert, P. S. 1997. Surface simplifi-
cation using quadric error metrics. In Proceedings of the
24th annual conference on Computer graphics and interac-
tive techniques, 209–216.
Gu´edon, A.; and Lepetit, V. 2024. Sugar: Surface-aligned
gaussian splatting for efficient 3d mesh reconstruction
and high-quality mesh rendering.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 5354–5363.
Guo, Y.; Yang, C.; Rao, A.; Liang, Z.; Wang, Y.; Qiao, Y.;
Agrawala, M.; Lin, D.; and Dai, B. 2023a.
Animatediff:
Animate your personalized text-to-image diffusion models
without specific tuning. arXiv preprint arXiv:2307.04725.
Guo, Y.; Zuo, X.; Dai, P.; Lu, J.; Wu, X.; Yan, Y.; Xu, S.; Wu,
X.; et al. 2023b. Decorate3d: text-driven high-quality texture
generation for mesh decoration in the wild.
Advances in
Neural Information Processing Systems, 36: 36664–36676.
Ke, B.; Obukhov, A.; Huang, S.; Metzger, N.; Daudt, R. C.;
and Schindler, K. 2024a. Repurposing diffusion-based im-
age generators for monocular depth estimation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 9492–9502.
Ke, B.; Obukhov, A.; Huang, S.; Metzger, N.; Daudt, R. C.;
and Schindler, K. 2024b. Repurposing Diffusion-Based Im-
age Generators for Monocular Depth Estimation. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR).
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3d gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.

<!-- page 9 -->
Landesbetrieb
Geoinformation
und
Vermessung
Hamburg.
2025.
3D-Geb¨audemodell
LoD3.0-HH.
https://metaver.de/trefferanzeige?docuuid=B438AD57-
223B-43A4-8E74-767CEC8A96D7.
CityGML dataset,
Freie und Hansestadt Hamburg, Datenlizenz Deutschland –
Namensnennung 2.0. Last updated April 2025.
Li, X.; Zhang, Y.; and Ye, X. 2024.
DrivingDiffusion:
Layout-Guided Multi-view Driving Scenarios Video Gener-
ation with Latent Diffusion Model. In European Conference
on Computer Vision, 469–485. Springer.
Li, Y.; Jiang, L.; Xu, L.; Xiangli, Y.; Wang, Z.; Lin, D.; and
Dai, B. 2023. Matrixcity: A large-scale city dataset for city-
scale neural rendering and beyond. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
3205–3215.
Li, Z.; Wang, Q.; Snavely, N.; and Kanazawa, A. 2022.
Infinitenature-zero: Learning perpetual view generation of
natural scenes from single images. In European Conference
on Computer Vision, 515–534. Springer.
Lin, C. H.; Lee, H.-Y.; Menapace, W.; Chai, M.; Siarohin,
A.; Yang, M.-H.; and Tulyakov, S. 2023. Infinicity: Infinite-
scale city synthesis. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, 22808–22818.
Liu, A.; Tucker, R.; Jampani, V.; Makadia, A.; Snavely, N.;
and Kanazawa, A. 2021. Infinite nature: Perpetual view gen-
eration of natural scenes from a single image. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, 14458–14467.
Liu, F.; Sun, W.; Wang, H.; Wang, Y.; Sun, H.; Ye, J.; Zhang,
J.; and Duan, Y. 2024. Reconx: Reconstruct any scene from
sparse views with video diffusion model.
arXiv preprint
arXiv:2408.16767.
Liu, Y.; Luo, C.; Tang, Z.; Li, Y.; Yang, Y.; Ning, Y.; Fan,
L.; Zhang, Z.; and Peng, J. 2025. TC-Light: Temporally Co-
herent Generative Rendering for Realistic World Transfer.
CoRR.
Lu, C.; Zhou, Y.; Bao, F.; Chen, J.; Li, C.; and Zhu, J. 2022.
Dpm-solver: A fast ode solver for diffusion probabilistic
model sampling in around 10 steps.
Advances in Neural
Information Processing Systems, 35: 5775–5787.
Lugmayr, A.; Danelljan, M.; Romero, A.; Yu, F.; Timofte,
R.; and Van Gool, L. 2022. Repaint: Inpainting using de-
noising diffusion probabilistic models. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, 11461–11471.
Luo, G.; Darrell, T.; Wang, O.; Goldman, D. B.; and Holyn-
ski, A. 2024. Readout guidance: Learning control from dif-
fusion features. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 8217–
8227.
Meng, C.; He, Y.; Song, Y.; Song, J.; Wu, J.; Zhu, J.-Y.;
and Ermon, S. 2021. Sdedit: Guided image synthesis and
editing with stochastic differential equations. arXiv preprint
arXiv:2108.01073.
Metzer, G.; Richardson, E.; Patashnik, O.; Giryes, R.; and
Cohen-Or, D. 2023. Latent-nerf for shape-guided generation
of 3d shapes and textures. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
12663–12673.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2021.
Nerf: Representing
scenes as neural radiance fields for view synthesis. Com-
munications of the ACM, 65(1): 99–106.
Peters, R.; Dukai, B.; Vitalis, S.; van Liempt, J.; and Stoter,
J. 2022. Automated 3D reconstruction of LoD2 and LoD1
models for all 10 million buildings of the Netherlands.
Poole, B.; Jain, A.; Barron, J. T.; and Mildenhall, B. 2022.
Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint
arXiv:2209.14988.
Ranftl, R.; Lasinger, K.; Hafner, D.; Schindler, K.; and
Koltun, V. 2020.
Towards robust monocular depth esti-
mation: Mixing datasets for zero-shot cross-dataset transfer.
IEEE transactions on pattern analysis and machine intelli-
gence, 44(3): 1623–1637.
Richardson, E.; Metzer, G.; Alaluf, Y.; Giryes, R.; and
Cohen-Or, D. 2023. Texture: Text-guided texturing of 3d
shapes. In ACM SIGGRAPH 2023 conference proceedings,
1–11.
Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-
mer, B. 2022. High-Resolution Image Synthesis With Latent
Diffusion Models. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
10684–10695.
Saharia, C.; Ho, J.; Chan, W.; Salimans, T.; Fleet, D. J.; and
Norouzi, M. 2022. Image super-resolution via iterative re-
finement. IEEE transactions on pattern analysis and ma-
chine intelligence, 45(4): 4713–4726.
SAVeNoW, P. 2020. LOD3 Road Space Models. https://
github.com/savenow/lod3-road-space-models.
Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Er-
mon, S.; and Poole, B. 2020. Score-based generative model-
ing through stochastic differential equations. arXiv preprint
arXiv:2011.13456.
Sun, W.; Chen, S.; Liu, F.; Chen, Z.; Duan, Y.; Zhang, J.; and
Wang, Y. 2024. Dimensionx: Create any 3d and 4d scenes
from a single image with controllable video diffusion. arXiv
preprint arXiv:2411.04928.
Wang, H.; Liu, Y.; Liu, Z.; Wang, W.; Dong, Z.; and Yang,
B. 2024a. VistaDream: Sampling multiview consistent im-
ages for single-view scene reconstruction. arXiv preprint
arXiv:2410.16892.
Wang, X.; Zhu, Z.; Huang, G.; Chen, X.; Zhu, J.; and Lu,
J. 2024b. DriveDreamer: Towards Real-World-Drive World
Models for Autonomous Driving. In European Conference
on Computer Vision, 55–72. Springer.
Wang, Z.; Lu, C.; Wang, Y.; Bao, F.; Li, C.; Su, H.; and Zhu,
J. 2023. Prolificdreamer: High-fidelity and diverse text-to-
3d generation with variational score distillation. Advances
in Neural Information Processing Systems, 36: 8406–8441.
Xie, H.; Chen, Z.; Hong, F.; and Liu, Z. 2024a. Citydreamer:
Compositional generative model of unbounded 3d cities. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, 9666–9675.

<!-- page 10 -->
Xie, H.; Chen, Z.; Hong, F.; and Liu, Z. 2024b.
Gaus-
sianCity: Generative Gaussian Splatting for Unbounded 3D
City Generation. arXiv preprint arXiv:2406.06526.
Xie, H.; Chen, Z.; Hong, F.; and Liu, Z. 2025.
City-
Dreamer4D: Compositional Generative Model of Un-
bounded 4D Cities. arXiv:2501.08983.
Yao, Z.; Chaturvedi, K.; and Kolbe, T. H. 2025. 3DCity-
DBV5: Open Source 3D City Database for CityGML. Chair
of Geoinformatics, Technical University of Munich. Version
5.0, https://www.3dcitydb.org/.
Yu, H.-X.; Duan, H.; Herrmann, C.; Freeman, W. T.; and
Wu, J. 2024a. WonderWorld: Interactive 3D Scene Genera-
tion from a Single Image. arXiv preprint arXiv:2406.09394.
Yu, H.-X.; Duan, H.; Hur, J.; Sargent, K.; Rubinstein, M.;
Freeman, W. T.; Cole, F.; Sun, D.; Snavely, N.; Wu, J.; et al.
2024b.
Wonderjourney: Going from anywhere to every-
where.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 6658–6667.
Yu, W.; Xing, J.; Yuan, L.; Hu, W.; Li, X.; Huang, Z.; Gao,
X.; Wong, T.-T.; Shan, Y.; and Tian, Y. 2024c. Viewcrafter:
Taming video diffusion models for high-fidelity novel view
synthesis. arXiv preprint arXiv:2409.02048.
Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding condi-
tional control to text-to-image diffusion models. In Proceed-
ings of the IEEE/CVF international conference on computer
vision, 3836–3847.
Zhao, G.; Wang, X.; Zhu, Z.; Chen, X.; Huang, G.; Bao, X.;
and Wang, X. 2024a. Drivedreamer-2: Llm-enhanced world
models for diverse driving video generation. arXiv preprint
arXiv:2403.06845.
Zhao, Y.; Zhou, Z.; Wang, Y.; Huang, J.; and Zhao, H.
2024b.
EasySynth: A Unified Toolkit for Generating
Multi-Modal Synthetic Data.
https://github.com/ydrive/
EasySynth. GitHub repository, accessed: 2025-08-01.

<!-- page 11 -->
VAE 
Encoder
Spatial Layout Condition
zero convolution
Reference Image
zero convolution
Unet
Encoder
Input
...
...
Control flow from
reference image
Figure 7: The architecture of ControlNet-N, the control flow
from reference image gives extra condition to ControlNet
7
Data Preparation and Implementation
City Scene Data.
Currently, there is a lack of publicly ac-
cessible high-quality datasets containing city mesh models
that can provide sufficient rendering data for training a Con-
trolNet. To evaluate the feasibility of our pipeline, we there-
fore leverage the detailed urban environment provided by the
City Sample Project (Epic Games 2022) in Unreal Engine
5 (Epic Games 2021), which features two distinct cities with
varied and realistic outdoor scenes. For training and evalu-
ation purposes, we generate distinct camera sequences and
render RGB images along with depth, normal, and semantic
maps using modified versions of tools from MatrixCity (Li
et al. 2023) and EasySynth (Zhao et al. 2024b). We set the
rendering resolution to 960×544 with a field of view (FoV)
of 45◦, simulating the front camera of a vehicle to ensure
good coverage of the forward view. The rendering interval is
set at 1 meter, with the camera consistently facing forward
horizontally. After this, we obtain 16k image with paired
control signals for training the ControlNets.
Implementation Details.
We utilize the training data de-
scribed above to train two ControlNets (Zhang, Rao, and
Agrawala 2023)—ControlNet-S and ControlNet-N—using
the frozen Stable Diffusion 1.5 (SD1.5)
(Rombach et al.
2022) as the base model. Additionally for ControlNet-N, the
extra control signal of warped image is randomly picked in
the range of 10-20 meters ahead of the current view. Both
networks are trained with a learning rate of 1e−5 and a batch
size of 128 for 10k iterations. During scene generation, the
next key frame is decided by thresholding the percentage of
incompleteness to be outpainted, which roughly results into
intervals of ∼20m distance with predefined threshold. For
long-range generation, the sequence is divided into blocks,
each consisting of 200 frames, which are processed through
our pipeline. We employ an autoregressive approach for con-
sistent generation, using the last generated frame from the
previous block as the starting point for the next. We perform
guided rectification on the latent for Ng = 100 iterations
with a learning rate lr = 0.00375 at each timestep.
8
Trial on Mesh Texturing Method –
Text2Tex
Several authors have attempted to adopt diffusion-based
generative models (Ranftl et al. 2020; Bhat et al. 2023;
Ke et al. 2024a; Saharia et al. 2022) to mesh texturing.
TEXTure(Richardson et al. 2023) and Text2Tex(Chen et al.
2023a) utilize depth-to-image diffusion models to texture
given meshes through inpainting. They tend to suffer from
visible seams and a gradual amplification of texture arti-
facts. Another family of methods like Fantasia3D(Chen et al.
2023b), Decorate3D (Guo et al. 2023b), SceneTex(Chen
et al. 2024) and Latent Paint(Metzer et al. 2023) is based on
Score Distillation Sampling (SDS) (Poole et al. 2022; Wang
et al. 2023), which is limited by the costly test-time opti-
mization and restricted to single objects or indoor scenes
with low polygon count. Moreover, these methods require
coverage of the relevant view field with training views.
Text2Tex (Chen et al. 2024) is a method for mesh textur-
ing based on depth-to-image generation models. It generates
and inpaints visible mesh texel for each predefined camera
views. At the end, a 3000 × 3000 RGB image is learnt to
represent mesh texture of objects with UV mapping. We at-
tempted to adopt it for outdoor scene mesh texturing. Unlike
Objaverse(Deitke et al. 2023), instances from City Sample
Project are high-poly meshes. To fit it into running code
of Text2Tex, we simplify a city block of 100 meters via
Quadric Edge Collapse Decimation(Garland and Heckbert
1997) and remove redundant parts except for visible mesh
faces from camera views (the second column in Fig. 8). For
the generation, We replace their inpainting diffusion mod-
ule with ours, and extend the texture on the mesh surface
by moving the camera backwards. Besides, we adopt a big-
ger size mesh texel with resolution 4096 × 4096 for better
quality. After all, several random views are picked to fur-
ther enhance the fidelity of the texture(Chen et al. 2024). As
you can see in Fig. 8, there’s apparent seams between each
inpainted part. Restricted by its application only on small
mesh models with low-poly, we could barely see its extend-
ability on large scale scene generation.
9
Implementation details of ControlNet-N
There is difference of processing the reference image com-
pared to other control signals(depth, semantic and normal
map). Similar to conditional latent diffusion model (Rom-
bach et al. 2022), the reference image is at first processed by
the VAE-Enconder and goes through several layers of con-
volution to align the feature dimension with other control
signal features. We illustrate this in Fig. 7.
10
Distributing Flattened Gaussian Surfels
On Mesh Surfaces
Our approach for surfel generation draws conceptual inspi-
ration from WonderWorld (Yu et al. 2024a), where depth
and surface normals are predicted for novel views. However,
since we directly utilize mesh geometry, we bypass this pre-
diction step by retrieving accurate depth and normal infor-
mation through rasterization.

<!-- page 12 -->
．．
！,.
.
 ．
 ＿
 
{
 
` ` r 
己：气，一｀ ｀＇三一二，．·｀··· , ', • ` · 
、
、, ',. ., .
Figure 8: Crossroad Mesh Model for SceneTex Training
Here, Gaussian surfels are parameterized with position p,
orientation in quaternion form q, scales in orthogonal direc-
tions s = [sx, sy, ϵ], opacity o and RGB color c. The Gaus-
sian kernel at any spatial position x is given by:
G(x) = exp

−1
2(x −p)T Σ−1(x −p)

,
(4)
where the covariance matrix Σ encodes the shape and orien-
tation of the surfel and is defined as:
Σ = Q · diag(s2
x, s2
y, ϵ2) · QT ,
(5)
with Q derived from the quaternion q. Our rendering
pipeline follows the same rasterization and alpha composit-
ing process as the 3D Gaussian Splatting (3DGS) frame-
work (Kerbl et al. 2023).
Given an image I of resolution H × W, we construct
H × W Gaussian surfels. For each surfel, the position p
is directly derived from its corresponding 3D position on
mesh, while it inherits color c from the pixel’s RGB value.
We assume that all surfaces are Lambertian, so c is treated
as view-invariant. To avoid rendering artifacts such as under-
sampling or holes when zooming in, the scales along x and y
axes are set as d/
√
2fx, d/
√
2fy, respectively. And ϵ is kept
small enough but still larger than zero to avoid numerical er-
rors. The opacity property is set to a constant 0.9. Similar
to position, we align the surfel’s normal with the mesh sur-
face normal at the corresponding pixel. Then we receive the
rotation matrix:
Qz = n,
Qx =
u × n
∥u × n∥,
Qy =
n × Qx
∥n × Qx∥,
(6)
with the global up-direction u = [0, 1, 0]T used to ensure a
consistent coordinate frame.
11
Stylized Rendering
The following prompts are leveraged for the inference of
Cosmos-Transfer1 (Alhaija et al. 2025) and TC-Light (Liu
et al. 2025) to achieve the visual results in the paper.
Full
prompt
for
cosmos
transfer1:
A
photorealistic scene of a quiet,
empty urban street on a sunny
day with some clouds in the sky.
The wide road with yellow center
lines stretches into the distance,
flanked by tall buildings with
classic architecture and street
lamps. Some building fronts feature
small potted shrubs, while a few
windowsills display potted flowers
or trailing leafy plants, adding
touches of greenery and charm to
the scene. The view is centered and
symmetrical, creating a peaceful
and cinematic atmosphere. Cool
moonlight and warm streetlamp glows
softly illuminate the buildings and
pavement, casting gentle shadows.
There are no people or vehicles,
enhancing the stillness. Urban
details like trash bins and cones
add realism. The composition draws
the eye toward a distant vanishing
point.
Full prompt for TC-Light: A photorealistic
depiction of a calm, empty urban
street at noon on a sunny day.
We also include extra stylized videos in the attachment,
they are generated by prompts:
Cosmos-transfer1
night
time
prompt:
A
photorealistic scene of a quiet,
empty urban street at night under

<!-- page 13 -->
a clear sky with scattered clouds.
The wide road with yellow center
lines stretches into the distance,
flanked by tall buildings with
classic architecture and street
lamps. Some building fronts feature
small potted shrubs, adding a touch
of greenery to the scene. The
view is centered and symmetrical,
creating a peaceful and cinematic
atmosphere. Cool moonlight and warm
streetlamp glows softly illuminate
the buildings and pavement, casting
gentle shadows. Urban details like
trash bins and cones add realism.
The composition draws the eye
toward a distant vanishing point.
TC-Light
night
time
prompt:
A
photorealistic depiction of a calm,
empty urban street at night under a
moonlit sky.
Cosmos-transfer1
rainy
day
prompt:
A
photorealistic scene of a quiet,
empty urban street during daytime
under a cloudy, rainy sky. The
wide road with yellow center
lines stretches into the distance,
flanked by tall buildings with
classic architecture. Rows of
street trees line both sides of the
road, their wet trunks and leaves
glistening slightly under the
diffused daylight. The rain-slick
asphalt reflects the buildings,
trees, and urban elements,
with scattered puddles creating
mirror-like surfaces. Soft daylight
filters through the overcast
sky, producing muted shadows and
emphasizing the textures of wet
pavement and facades. Urban details
like trash bins and traffic cones
add realism. The composition is
centered and symmetrical, guiding
the eye toward a distant vanishing
point softened by mist and rain.
Cosmos-transfer1
snowy
day
prompt:
A
photorealistic scene of a quiet,
empty urban street during daytime
in snowy weather. The wide road
stretches into the distance,
flanked by tall buildings with
classic architecture and street
lamps dusted with snow. Trees
lining the sidewalks are covered
with snow-laden branches, adding
to the serene winter atmosphere.
Some building fronts feature
small potted shrubs, adding
subtle touches of greenery amid
the white landscape. The view
is centered and symmetrical,
creating a peaceful and cinematic
atmosphere. Soft daylight reflects
off the snow-covered pavement and
buildings, casting diffuse shadows.
Urban details like trash bins and
cones partially covered in snow
add realism. The composition draws
the eye toward a distant vanishing
point.
