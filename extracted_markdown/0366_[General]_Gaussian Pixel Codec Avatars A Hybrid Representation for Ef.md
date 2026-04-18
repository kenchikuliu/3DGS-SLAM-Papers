<!-- page 1 -->
Gaussian Pixel Codec Avatars: A Hybrid Representation for Efficient Rendering
Divam Gupta3*, Anuj Pahuja1, Nemanja Bartolovic1, Tomas Simon1, Forrest Iandola2, Giljoo Nam1
1Meta Codec Avatars Lab
2Meta Reality Labs
3Stellon Labs
(a)  Textured Mesh
(b)  3D Gaussians
(c)  Hybrid Rendering
+
Figure 1. Gaussian Pixel Codec Avatars. We present Gaussian Pixel Codec Avatars (GPiCA), photorealistic head avatars that combine
(a) a textured triangle mesh, with (b) 3D Gaussians, to produce a (c) hybrid rendering representation that retains high realism while being
efficient to render even on low-end devices
Abstract
We present Gaussian Pixel Codec Avatars (GPiCA), pho-
torealistic head avatars that can be generated from multi-
view images and efficiently rendered on mobile devices.
GPiCA utilizes a unique hybrid representation that com-
bines a triangle mesh and anisotropic 3D Gaussians. This
combination maximizes memory and rendering efficiency
while maintaining a photorealistic appearance.
The tri-
angle mesh is highly efficient in representing surface ar-
eas like facial skin, while the 3D Gaussians effectively han-
dle non-surface areas such as hair and beard. To this end,
we develop a unified differentiable rendering pipeline that
treats the mesh as a semi-transparent layer within the vol-
umetric rendering paradigm of 3D Gaussian Splatting. We
train neural networks to decode a facial expression code
into three components: a 3D face mesh, an RGBA texture,
and a set of 3D Gaussians. These components are rendered
simultaneously in a unified rendering engine. The networks
are trained using multi-view image supervision. Our re-
sults demonstrate that GPiCA achieves the realism of purely
Gaussian-based avatars while matching the rendering per-
formance of mesh-based avatars.
*Work done at Meta.
1. Introduction
Modeling, animating, and rendering human faces have been
an important topic of interest in computer graphics for many
years [5, 29]. Recently, there have been remarkable ad-
vancements in creating animatable head avatars [15, 32,
33, 39].
These achievements can be attributed to three
key components: large-scale face datasets, learning-based
approaches, and efficient representations.
Firstly, multi-
view video capture systems [19, 40] have enabled the col-
lection of extensive datasets of faces in motion, allowing
for fully data-driven methods that can model subtle expres-
sions. Secondly, deep generative models such as GAN [9]
and VAE [18] have made it possible to learn compact la-
tent spaces of geometry and appearance of faces in mo-
tion.
Lastly, advances in scene representations, such as
Neural Radiance Field (NeRF) [27] and 3D Gaussian Splat-
ting (3DGS) [16], have provided the means for photorealis-
tic modeling and rendering of avatars. This paper focuses
on the representation aspect of head avatars and proposes a
new hybrid representation combining triangle meshes and
3D Gaussians for efficient rendering.
Why does representation matter? First, for head avatars,
where accurately capturing shape and appearance is essen-
tial, the type of representation being used can limit expres-
siveness.
Triangle meshes are a popular choice in com-
puter graphics due to their simplicity and efficient render-
1
arXiv:2512.15711v1  [cs.CV]  17 Dec 2025

<!-- page 2 -->
z
Θv
Dm
Hybrid Render
3D Gaussians
Sampling Mask
Dg
Share parameters
Tꭤ
Tc
X
t, R, s, o
c
Mesh Depth
Rasterized Opacity
Rasterized Texture
Differentiable Hybrid Renderer
Mesh Geometry
GT
Reconstruction Loss
1st Pass
2nd Pass
Figure 2. Overview. Given an expression latents code and view direction, our model outputs 3D Gaussians and a mesh in the UV texture
space. The Gaussian positions are predicted relative to the mesh. A sampling mask is used to select 3D Gaussians for key regions like
hair, beard etc. In the first pass, the mesh is rasterized to produce RGB texture, opacity and depth. In the second pass the 3D Gaussians,
rasterized texture, depth and opacity are sent to our differentiable hybrid renderer. The renderer first accumulates colors of Gaussians in
front of the mesh, then accumulate the mesh component and finally accumulates Gaussians behind the mesh. The system is trained jointly.
ing. However, it is challenging to accurately reconstruct
images of hair into a mesh. Second, representations directly
impact rendering performance. This is crucial for real-time
AR/VR applications like telepresence, where latency has to
be minimal while being compute limited. Therefore, we
aim to find a representation that is efficient enough to run
on mobile devices while maintaining high fidelity in repro-
ducing complex human face geometry and appearance.
We present Gaussian Pixel Codec Avatars (GPiCA), pho-
torealistic head avatars learned from multi-view images.
The gist of GPiCA is a hybrid representation of a trian-
gle mesh and anisotropic 3D Gaussians that maximizes the
memory and rendering efficiency while achieving photore-
alistic appearance, especially around hair. The mesh can ef-
ficiently represent surface areas like facial skin whereas 3D
Gaussians deal with non-surface areas like hair and beard.
To this end, we develop a unified differentiable rendering
pipeline that views the mesh as a semi-transparent layer
within the volumetric rendering paradigm of 3DGS. With
that, we train neural networks that decode a facial expres-
sion code into a 3D face mesh, a RGBA texture, and a set of
3D Gaussians. All the three components are rendered in a
unified rendering engine at once. The networks are trained
via multi-view image supervision. We show that GPiCA
is as efficient as mesh-based avatars in terms of rendering
performance and is as realistic as purely Gaussian-based
avatars.
2. Related Work
2.1. Representations for Face Modeling
Mesh
The triangle mesh is one of the most widely used
representations for surfaces in computer graphics, and mod-
ern GPUs can rasterize billions of triangles per second. In
addition to its exceptional computational efficiency, meshes
also provide a strong benefit to face modeling; the shared
topology enables to build dense correspondences across dif-
ferent identities [1, 8]. This has been shown in the suc-
cess of various 3D morphable face models (3DMM), most
of which are based on triangle meshes [6]. Such facial 3D
template models [21] have been used to create photorealis-
tic mesh-based head avatars using various datasets such as
selfies [10], in-the-wild videos [17], or multi-view face cap-
tures [23, 26]. While being efficient and robust, meshes are
reported to have limited capabilities to faithfully reproduce
the complex appearance of hair.
Volume
Volumes are good alternatives to meshes for non-
surface areas like hair thanks to its flexibility to learn ge-
ometry and appearance from images. It has been shown
that volumes are effective representations for learning 3D
faces from unstructured large-scale portrait images [2, 35]
or creating animatable head avatars from structured multi-
view face captures [24]. Volumetric representations are of-
ten combined with mesh-based 3DMMs for the sake of face
registration [47], tracking [46], and animation [25]. While
2

<!-- page 3 -->
volumes usually achieve noticeable visual improvements in
the areas like hair, beard, and eyebrow, its high computa-
tional cost can be a problem in realtime applications with
a budget constrained environment such as mobile AR/VR
devices.
3D
Gaussian
Splatting
3D
Gaussian
Splatting
(3DGS) [16] employs anisotropic 3D Gaussians as a
scene representation and achieves realtime rendering
performance via software rasterization. A number of works
have demonstrated the effectiveness of 3DGS as a repre-
sentation of head avatars [31, 32, 34, 41, 43, 45]. Similar
to volumetric representations, binding 3D Gaussians with a
3DMM is a common practice for registering, tracking, and
animating faces [31, 32, 43, 45]. Despite being an efficient
representation for realtime applications, 3DGS still has a
trade-off between quality and memory/speed; millions of
3D Gaussians are typically required to achieve high fidelity
photorealism [7, 13, 20, 28]. Although high-end GPUs can
render millions of 3D Gaussians in realtime, achieving this
within budget constrained environments, such as mobile
AR/VR devices, remains a research challenge.
Meshes and Gaussians
Concurrent work to ours ex-
plored a hybrid representation of a mesh and 3D Gaussians
that both representations contribute to final pixel colors.
Wang et al. [38] explicitly divided head avatars into two
regions, face and hair, and used mesh for face and 3DGS
for hair. Unlike our work, they do not jointly train mesh
and 3D Gaussians. They also treat mesh as a fully opaque
layer, and they composite the mesh and 3DGS renderings
via screen space alpha blending. Xiao et al. [42] presented
a hybrid Gaussian-mesh rendering technique that treats the
mesh as a semi-transparent layer within 3DGS rasterization
framework, similar to our hybrid rendering paradigm. They
focus on a dynamic scene reconstruction problem and do
not focus reducing the number of Gaussians. Our primary
goal is to build a head avatar that can be animated from var-
ious driving signals such as head-mount cameras in mobile
VR headsets which have compute constraints.
Svitov et al. [36] proposed a hybrid method for gener-
ating full body avatars from monocular videos. First, they
separately generate a mesh and a Gaussian avatar. Then in
the filtering stage, they remove the 3D Gaussians which are
not needed. They always keep the mesh opaque and only
update the color and opacity of the 3D Gaussians in the fil-
tering stage. Because of this, the Gaussians are not able to
capture details in complex areas like hair where 3D Gaus-
sians are needed the most. On the other hand, we jointly
train a semi-transparent mesh and 3D Gaussians, because of
which we are able to capture those complex areas. Our ex-
periments show that mesh coarsely approximates areas like
hair, and having an opaque mesh hinders the 3D Gaussians.
Figure 3. Learned mesh and Gaussian primitives. Top row:
Per-vertex normals of the learned mesh. Middle row: Learned
Gaussians as 3D ellipses. Bottom row: Final renders.
2.2. Meshes and Gaussians for 3D Reconstruction
There have been efforts to relate 3D Gaussians to meshes
for 3D reconstruction. A common approach is to put a hard
constraint that forces 3D Gaussians to lie on the faces of a
triangle mesh and update the mesh vertices during optimiza-
tion processes [3, 11, 22, 37]. Or one can put a soft con-
straint that allows some thickness on a surface, so the Gaus-
sians can better represent volumetric appearance [12]. It
was also shown that directly optimizing 2D oriented Gaus-
sian disks, instead of 3D Gaussians, helps better surface re-
construction [4, 14]. All these works either use meshes as
a hidden representation to support the optimization of 3D
Gaussians or simply extract meshes after the optimization,
whereas our hybrid mesh-Gaussian rendering allows both
representations to contribute to the final pixel color.
3. Methodology
Our method combines a triangle mesh and anisotropic 3D
Gaussians to render an avatar. The mesh captures the ma-
jority of the avatar’s visual and geometrical information,
while the 3D Gaussians supplement the visual data that the
mesh cannot represent. These 3D Gaussians are strategi-
cally placed in areas such as head hair, facial hair, eyebrows,
and eyelashes, which are beyond the representational capac-
ity of a mesh. To render the Gaussian with a mesh, we first
rasterize the mesh using differentiable rendering. Follow-
ing this, we blend the visual information with the 3D Gaus-
sians in 3D space. We refer to this as hybrid mesh-Gaussian
rendering. This process is differentiable, allowing it to be
3

<!-- page 4 -->
learned in an end-to-end manner.
Our objective is not to reconstruct a scene, but to create
an animatable avatar that can be rendered on mobile de-
vices. To this end, we train a variational autoencoder that
produces all the necessary visual and geometric informa-
tion. The encoder produces a latent code z which serves as
a conditioning input for the decoder to produce the avatar.
To make our rendering efficient on mobile devices, only use
3D Gaussians where they are absolutely needed. While our
method is independent of the underlying mesh model, we
utilize PiCA [26] due to its efficiency and high-quality mesh
rendering on mobile devices.
3.1. Data Acquisition and Preprocessing
Our objective is to develop a model capable of generating
3D renderings of an avatar. Hence, in line with previous
works [26, 32], we utilize multi-view images captured from
approximately 100 cameras. Each subject is recorded per-
forming a predefined set of facial expressions, gaze mo-
tions, and sentences. Following the approach in [26, 32],
we conduct topologically consistent coarse mesh tracking
and per-frame unwrapped averaged textures using the multi-
view images.
3.2. Encoder
The encoder takes in the coarse mesh and the unwrapped
averaged texture as input and produces a latent code z. We
use a network architecture similar to [26]. We only use this
encoder during training since computing tracked meshes re-
quire high quality multi-view images.
3.3. Decoder
To make the avatar model animatable, we use a decoder Dm
to produce mesh and texture, and a decoder Dg to produce
3D Gaussians. Both decoders are conditioned on the latent
code. The decoders use a fully convolutional architecture
similar to PiCA [26], for its efficiency to run on mobile de-
vices. We produce both the mesh and Guassian parameters
in the same UV space so that we can share compute and
features across the convolutional layers in Dm and Dg.
3.4. Mesh Geometry
The mesh provides the avatar’s underlying geometry and
visual information and guides the 3D Gaussians which are
placed on it. We use a semi-transparent mesh to render the
mesh with 3D Gaussians jointly. Although our method is
independent of the underlying mesh model, we base our
model on PiCA [26] for our experiments due to its abil-
ity to produce high-quality meshes and to run efficiently on
mobile devices.
The mesh decoder Dm takes the latent code z and view-
ing direction Θv as inputs, generating a texture and a ge-
ometry. All the outputs of the decoder are in the UV texture
space producing a K ×K map. The decoder outputs a color
texture map Tc ∈RK×K in UV space, an opacity texture
map Tα ∈RK×K, and a position map that contains vertices
X = {x1, x2 . . . xK2} at each UV location, as follows:
{Tc; Tα; X} = Dm(z; Θv).
(1)
The color texture decoder is conditioned on both the latent
code and the viewing direction which helps the model in
generating view-dependent appearance. However, the ge-
ometry and transparency map are independent of the view-
ing angle, and hence, they are conditioned solely on the la-
tent code.
The position map produced by the decoder along with
a predefined human face topology τ, is used to construct
a mesh of the face. This mesh is rendered using a differ-
entiable renderer, allowing us to learn the whole system in
an end-to-end manner. For a pixel p in the rendered image,
the mesh color C′
p, mesh opacity α′
p, and mesh depth d′
p
are produced by rasterizing the color texture Tc and opacity
texture Tα,
{C′
p, α′
p, d′
p} = rasterize(X, τ, {Tc, Tα}, p).
(2)
Note that, in this stage, we render this mesh as an opaque
surface with no blending, and therefore {C′
p, α′
p, d′
p} cor-
respond to the color, opacity, and depth of the front-most
surface. We also use edge gradients [30] for improved op-
timization of the mesh.
3.5. Gaussian Splatting
Mesh-based models like PiCA [26] can create most facial
features, but they struggle to produce high-quality hair, in-
cluding head hair, facial hair, eyelashes, and eyebrows. On
the other hand, 3D Gaussian Splatting is proficient at rep-
resenting these elements. Hence, we use a set of 3D Gaus-
sians along with the mesh to improve the quality in those
regions.
For a set of N 3D Gaussians, each individual
Gaussian, denoted as gk = {tk, Rk, sk, o, ck}, where tk
is the relative position, Rk ∈SO(3) is the rotation, sk is
the scale, ok is the opacity, and ck is the color.
We generate the 3D Gaussians by a decoder Dg that
takes the latent code as input, and we share the convolu-
tional neural network from the mesh decoder to produce the
3D Gaussian properties. Hence, the 3D Gaussians are pro-
duced in the same UV space, allowing feature re-use and
shared computations.
{g1, g2 . . . gN} = Dg(z; Θv)
(3)
Since the mesh already captures the underlying geometry,
we represent the positions of the 3D Gaussians relative to
the mesh. The decoder generates the differences in posi-
tions with respect to the mesh coordinate positions.
4

<!-- page 5 -->
(a) GS 16k
(b) GS 65k
(c) Ours 16k
(d) GT
Figure 4. Comparing with vanilla Gaussian avatars. With only
16k Gaussians, vanilla Gaussian avatars struggle to capture facial
details. In contrast, our hybrid avatars, also using 16k 3D Gaus-
sians, achieve significantly sharper representations. They are com-
parable to vanilla Gaussian avatars with 65k 3D Gaussians which
are much slower to render.
To enhance rendering efficiency, we predict color di-
rectly from the decoder instead of using spherical harmon-
ics. The predicted color is conditioned on the viewing an-
gle and the latent code, helping in the production of view-
dependent appearance.
We have limited budget on total number of 3D Gaus-
sians that can be rendered efficiently on low-end devices.
Hence, we want more 3D Gaussians in regions like hair,
which cannot be represented by the mesh. For that, we use
a static mask which selects 16,348 3D Gaussians from the
256×256 UV space. Out of the selected 3D Gaussians, 75%
of them are selected from the hair regions and the rest 25%
are selected from other regions. To compute the UV mask
for the hair, we use semantic segmentation on the images
and project them to the UV map.
3.6. Hybrid Mesh-Gaussian Renderer
After producing the mesh and the 3D Gaussians from the
decoder, we jointly render them using our hybrid Mesh-
Gaussian renderer. We use a two-pass approach where the
mesh is first rendered separately, and then blend that with
the Gaussians in 3D.
For each 3D Gaussian gk, let the depth for that gaussian
in the camera space be dk. We use the 3D Gaussians in the
sorted order, and d1 ≤d2 · · · ≤dN.
Figure 5. Mesh and Gaussian opacity contribution. First row:
Mesh is colored green and Gaussians are colored blue. Second
row: RGB contribution from the mesh. Third row: RGB contribu-
tion from the Gaussians. Fourth row: Final renders.
(a) PiCA
(b) Ours 16k
(c) GT
Figure 6. Comparing with mesh based avatars. PiCA can only
capture flat surfaces and struggles with complex areas like hair,
where it coarsely approximates volumetric details. Our Hybrid
GS avatars overcome this limitation by adding small number of
3D Gaussians.
For every pixel p, we first accumulate all the 3D Gaus-
sians which are in front of the mesh, then we accumulate
the mesh color of that pixel, and finally accumulate the rest
5

<!-- page 6 -->
of the 3D Gaussians.
If the depth of the mesh at pixel p is d′
p, the color contri-
bution of Gaussians in front of the mesh will be computed
by accumulating g1, g2 . . . gm−1, where d1 ≤d2 · · · ≤
dm−1 < d′
p,
Cfront =
m−1
X
k=1
ckαk
k−1
Y
j=1
(1 −αj) ,
(4)
where the transparency αk is evaluated using the 2D covari-
ance of the Gaussian and multiplied by the per-Gaussian
opacity ok.
The color contribution of the mesh part will be computed
using the color and opacity of the mesh at p and the trans-
mittance of the 3D Gaussians in front of it.
Cmesh = C′
pα′
p
m−1
Y
j=1
(1 −αj)
(5)
The color contribution of Gaussians that are behind the
mesh is computed by accumulating gm, gm+1 . . . gN, where
d′
p ≤dm ≤dm+1 · · · ≤dN,
Cbehind = (1 −α′
p)
N
X
k=m
ckαk
k−1
Y
j=1
(1 −αj) .
(6)
The final color at pixel p is then
Cp = Cfront + Cmesh + Cbehind
(7)
This makes it differentiable and lets us back propagate gra-
dients to the mesh and Gaussians. By jointly training the
decoder with the hybrid renderer, the network can decide
what to represent using Gaussians and what to represent
using mesh. Fig. 5 shows how the mesh tends to explain
flatter surfaces, like the skin, whereas the Gaussians tend to
explain structures like hair and eyelashes.
3.7. Optimization and regularization
Given multi-view video data of a person, we jointly op-
timize all trainable network parameters.
We use mean
squared error loss between the rendered color Cp and and
ground truth color CGT
p
on all the pixels of the image. The
latent code z is computed from an encoder E . E takes the
tracked mesh and an average texture computed by back-
projecting ground truth images onto the tracked mesh as in-
put. E and the decoders Dg and Dm are trained jointly with
a Kullback-Leibler divergence loss on the encoder outputs.
We regularize the Gaussians with a scale loss [32] which
penalizes if the scale term sk is not in a given range,
Ls = mean(ls), ls =





1/ max(s, 10−7)
if s < 0.1
(s −10.0)2
if s > 10.0
0
otherwise.
(8)
Subject
Method
MAE
SSIM
PSNR
LPIPS
Subject 1
PiCA
9.13
0.59
26.05
0.45
GS 16k
8.15
0.63
27.24
0.49
GS 65k
7.67
0.68
27.85
0.37
Ours 16k
7.65
0.66
27.68
0.37
Subject 2
PiCA
6.48
0.67
28.76
0.45
GS 16k
6.37
0.7
29.3
0.52
GS 65k
5.8
0.72
30.02
0.42
Ours 16k
5.7
0.71
30.12
0.39
Subject 3
PiCA
6.59
0.68
28.18
0.39
GS 16k
6.65
0.67
28.53
0.51
GS 65k
6.29
0.7
28.96
0.4
Ours 16k
6.13
0.71
29.06
0.35
Subject 4
PiCA
6.72
0.67
28.13
0.44
GS 16k
6.53
0.68
28.77
0.55
GS 65k
6.79
0.7
28.77
0.43
Ours 16k
6.04
0.7
29.25
0.39
Subject 5
PiCA
7.33
0.66
27.54
0.4
GS 16k
6.88
0.69
28.37
0.48
GS 65k
6.48
0.72
28.95
0.36
Ours 16k
6.21
0.72
29.18
0.33
Table 1. Quantitative evaluation on face dataset. Our proposed
hybrid approach with 16k Gaussians outperforms PiCA (a pure
mesh approach) and vanilla GS with 16k Gaussians, and is com-
parable to vanilla GS with 65k Gaussians.
8k
16k
32k
65k
LPIPS
Hybrid GS
0.36
0.33
0.31
0.31
Vanilla GS
0.56
0.48
0.42
0.36
MAE
Hybrid GS
6.41
6.21
5.92
5.91
Vanilla GS
7.27
6.88
6.66
6.48
PSNR
Hybrid GS
28.82
29.18
29.56
29.57
Vanilla GS
27.92
28.37
28.67
28.95
Table 2. Quality metrics with respect to number of Gaussians.
This table shows LPIPS, MAE, and PSNR metrics for hybrid and
vanilla GS models for varying numbers of Gaussian splats.
We also ensure that the Gaussians are close to the mesh by
using a loss to penalize if translation vector tk is very large,
Lt = mean(lt), ls =
(
∥t∥−10.0
if ∥t∥> 10.0
0
otherwise.
(9)
Additionally, to regularize the generated mesh, we use a
normal loss and a Laplacian smoothness regularization.
Please refer to PiCA [26] for more details for the mesh reg-
ularization terms.
6

<!-- page 7 -->
4. Experiments
4.1. Experimental Setting
We evaluate our method on a dataset of human faces con-
taining 5 subjects with diverse genders and hairstyles. We
use 5 held-out camera poses that were not used during train-
ing to generate our qualitative and quantitative results. We
used around 2,000 video frames of size 2048 × 1334 for
training and evaluation. These frames contain several hu-
man expressions. We report MAE, PSNR, LPIPS [44], and
SSIM on the face region of the image. For a fair compari-
son, we train and evaluate all the compared models with the
same quantity of data and the same number of iterations,
and use the same base decoder architecture. Additionally,
we also evaluate our method on a similar full body image
dataset of a single subject.
4.2. Baselines
We compare our model with a mesh-based PiCA [26]
model, and purely 3D Gaussian Splatting based avatar mod-
els.
The only difference between our hybrid model and
the baseline Gaussian Splatting model is the absence of
the mesh component in the rendering.
Our GPiCA hy-
brid model uses 16,384 3D Gaussians, hence we compare
against 16,384 and 65,536 (4×) Gaussians vanilla 3D Gaus-
sian Splatting based models. While even more Gaussians
can be used to represent these avatars for better quality, we
find that it gets prohibitively more expensive for low-end
devices with integrated GPUs to render them at very high
frame rates. We use the same network architecture for the
decoders that produce the per-frame mesh and/or Gaussian
primitives in all the models for a fair comparison. We also
compare our method with opaque mesh hybrid Gaussian
Splatting which is used by works such as [36, 38].
4.3. Qualitative Results
Fig. 4 and 6 show renders of our model and the baselines
from camera poses that are unseen during training. We can
see that the mesh-only PiCA model fails to accurately rep-
resent hair and non-flat regions. The mesh-only model also
produces sharp edges which makes it easily distinguishable
from reality. Due to limited number of 3D Gaussians, the
purely 3D Gaussian Splatting based model is not able to
represent sharp details in face texture (skin spots, beard
etc.), something which the PiCA model is better at, while
being faster. On the other hand, our GPiCA hybrid model
does not have the shortcomings of the baselines while using
a limited number of 3D Gaussians. GPiCA learns detailed
skin textures, non-flat regions like hair, and also does not
have any sharp edge artifacts from the mesh – producing
images closer to the ground truth. Other works, such as
[36, 38], use an opaque mesh for blending with 3D Gaus-
sians. Figure 7 shows that our proposed hybrid avatars per-
form better in regions like hair, which need complex volu-
metric modeling. Hybrid 3D Gaussians with opaque mesh
fail to model intricate volumetric details because coarse
meshes tend to hinder 3D Gaussians that could go behind
them, but our method does not have this limitation.
4.4. Quantitative Results
Table 1 shows quantitative results of our model and its com-
parison with other baseline models. We can see that our hy-
brid model has better LPIPS and MAE scores compared to
other baselines for all subjects, and better SSIM and PSNR
scores for most subjects. Our hybrid model with 16,384 3D
Gaussians is able to outperform vanilla 3DGS based models
with same number of Gaussians and PiCA in all cases. De-
spite having 1/4th the number of 3D Gaussians, our model
is able to get better LPIPS scores compared to the vanilla
3DGS based models with 65,536 3D Gaussians and per-
forms competitively on other metrics. The results show that
we can reduce the required number of Gaussians by 4× and
effectively leverage mesh-based model for a lot of the heavy
work, and still get similar or better quality outputs.
4.5. Rendering Runtime
We evaluate the mean GPU runtime performance of ren-
dering PiCA, vanilla 3DGS models, and our hybrid GPiCA
models in Table 3. We implemented efficient Vulkan based
rendering modules for each of the models that run on a stan-
dard Quest 3 VR headset equipped with Adreno 740 GPU
and a Hexagon Tensor Processor (HTP). The render res-
olution was kept to be 2048 × 1334 for all the numbers
and Subject 5 models were used. We run the decoder on
the HTP and the rendering on the GPU. The decoder for
PiCA takes 2.2 ms, and the decoder for GS and hybrid mod-
els takes 6.9 ms. For PiCA, the rendering time includes
for mesh rasterization and the multi-layer perceptron that
runs in screen space in the fragment shader. For 3DGS,
we implement a compute based rendering pipeline using
Vulkan compute shaders. For GPiCA hybrid renderer, we
modify the final color accumulation compute shader in the
3DGS pipeline to also take RGBA and depth buffers from
the PiCA renderer as inputs. We then use these additional
inputs for computing the final per-pixel color and opacity
values from both the rasterized mesh and the Gaussians as
described in Eq. 7. Our hybrid renderer runtime is in the
same range as the time for running PiCA renderer followed
by 3DGS renderer, but the quality exceeds that of the 3DGS
model with 4× Gaussians, the latter being much more ex-
pensive to render natively on VR devices with high refresh
rates (>72 Hz).
7

<!-- page 8 -->
Models
LPIPS↓
Render Time (ms)
PiCA (Mesh)
0.4
1.633
GS 16k
0.48
9.0997
GS 65k
0.36
19.266
Hybrid GS 16k
0.33
10.900
Table 3. Performance and runtimes on Quest 3. This table
shows the LPIPS error metric and corresponding rendering times
for various methods on a Quest 3 mobile GPU. Note that the Hy-
brid GS model with 16k splats has comparable or improved per-
formance with a vanilla GS model of 65k splats with significantly
improved runtime.
4.6. Further Analysis
4.6.1. Effect of number of Gaussians
Fig. 8 and Table 2 show how the LPIPS scores of vanilla
GS and our hybrid GS models change with the number of
3D Gaussians for Subject 5. We can see that LPIPS reduces
when we increase the number of 3D Gaussians. Our hybrid
GS based model is always better than the vanilla 3DGS base
model for the same number of Gaussians, and better than the
models with 2×/4× Gaussians. We chose 16,384 Gaussians
for our hybrid GS based model for most of the experiments
as it is a good balance between runtime speed and quality.
4.6.2. Visualizing Geometry
Fig.
3 shows the visualizations of the geometry of the
learned mesh and the 3D Gaussians for our hybrid GS
model. We can see that the 3D Gaussians become long and
elongated at the hair region learning finer hair details. The
mesh geometry is smooth at the non-hair regions of the face
and captures a coarse shape of the hair regions. Fig 5 shows
a visualization of our hybrid GS model, highlighting the
contributions of the mesh part (highlighted in green color)
compared with the contributions from the Gaussians (high-
lighted in blue color). It also shows the learned mesh and
Gaussians rendered independently. We can see how Gaus-
sians complement the mesh at hair regions and composite
together to generate the final colors.
4.7. Ablations
Hybrid GS with uniform initialization. Rather than the
proposed initialization where we put more 3D Gaussians in
the hair region using segmentations, we initialize the loca-
tions of 3D Gaussians uniformly over the UV map.
Hybrid GS with opaque mesh. Rather than the proposed
method where the opacity of the mesh surface layer is pre-
dicted by the decoder, we just set treat the mesh as fully
opaque and set the opacity to 1.0.
We show the performance metrics for both the above ab-
lations in Table 5 and observe a decrease in performance,
which is expected. Opaque mesh would mean that the mesh
colors are all accumulated before the Gaussians, which isn’t
(b) Opaque Hybrid 16k 
(c) Ours 16k
(a) GT
Figure 7. Comparing with opaque hybrid avatars. Although
opaque mesh hybrids show enhancements over pure mesh avatars,
they still have limitations in complex regions like hair, where the
mesh component coarsely approximates fine volumetric details.
Our proposed hybrid avatars do not face these limitations.
10000
20000
30000
40000
50000
60000
Number of 3D Gaussians
0.0
0.1
0.2
0.3
0.4
0.5
0.6
LPIPS Score
Hybrid GS
Vanilla GS
Figure 8. Number of Gaussians vs. LPIPS. We show LPIPS
reconstruction error for hybrid and vanilla GS models, for varying
number of Gaussians. Note that a hybrid model with 8k splats
performs as well as a vanilla GS model with 65K splats on this
data.
Model Name
MAE
SSIM
PSNR
LPIPS
GS 260K
3.15
0.72
26.78
0.17
PiCA
2.84
0.73
26.56
0.23
Opaque Hybrid 16k
3.04
0.71
26.24
0.17
Ours 16k
2.85
0.73
26.9
0.15
Table 4. Quantitative evaluation on full body dataset. Our pro-
posed hybrid approach with 16k 3D Gaussians outperforms vanilla
Gaussian avatars while using 16x fewer 3D Gaussians. We also
outperform PiCA and opaque mesh hybrid avatars.
necessarily true for avatars.
With opaque mesh, it also
fails to model intricate areas such as hair because the mesh
coarsely approximates thin structures. Uniform initializa-
tion of 3D Gaussians still performs well but makes it rel-
atively more difficult to displace the Gaussians where they
are needed the most.
8

<!-- page 9 -->
Method
MAE SSIM PSNR
LPIPS
Hybrid GS (Opaque)
6.591
0.688 28.679
0.374
Hybrid GS (Uniform)
6.432
0.692 28.951
0.38
Hybrid GS
6.347
0.699 29.058
0.367
Table 5. Ablations averaged across all 5 subjects. In Hybrid
GS Uniform we do not use a sampling mask to initialize more
3D Gaussians at hair regions. In Hybrid GS Opaque, we use an
opaque mesh rather than semi-transparent mesh.
5. Conclusion
We present Gaussian Pixel Codec Avatars, a hybrid ap-
proach to decode and render avatars, leveraging the effi-
ciency of mesh and the expressivity of 3D Gaussian Splat-
ting in a unified pipeline. Unlike opaque mesh hybrid Gaus-
sian Splatting approaches, our method is able to model thin
and complex regions like hair, which is very important for
avatars. The hybrid renderer makes it possible to render
very high quality avatars on low-compute, low-latency de-
vices like VR headsets at their native refresh rates.
References
[1] Chen Cao, Yanlin Weng, Shun Zhou, Yiying Tong, and Kun
Zhou. Facewarehouse: A 3d facial expression database for
visual computing. IEEE Transactions on Visualization and
Computer Graphics, 20(3):413–425, 2013. 2
[2] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano,
Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J
Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient
geometry-aware 3d generative adversarial networks. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 16123–16133, 2022. 2
[3] Hanlin Chen, Chen Li, and Gim Hee Lee. Neusg: Neural im-
plicit surface reconstruction with 3d gaussian splatting guid-
ance. arXiv preprint arXiv:2312.00846, 2023. 3
[4] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu,
Huamin Wang, and Weiwei Xu.
High-quality surface re-
construction using gaussian surfels.
In SIGGRAPH 2024
Conference Papers. Association for Computing Machinery,
2024. 3
[5] Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter
Duiker, Westley Sarokin, and Mark Sagar.
Acquiring the
reflectance field of a human face.
In Proceedings of the
27th annual conference on Computer graphics and interac-
tive techniques, pages 145–156, 2000. 1
[6] Bernhard Egger, William AP Smith, Ayush Tewari, Stefanie
Wuhrer, Michael Zollhoefer, Thabo Beeler, Florian Bernard,
Timo Bolkart, Adam Kortylewski, Sami Romdhani, et al.
3d morphable face models—past, present, and future. ACM
Transactions on Graphics (ToG), 39(5):1–38, 2020. 2
[7] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, De-
jia Xu, and Zhangyang Wang. Lightgaussian: Unbounded
3d gaussian compression with 15x reduction and 200+ fps.
arXiv preprint arXiv:2311.17245, 2023. 3
[8] Thomas Gerig, Andreas Morel-Forster, Clemens Blumer,
Bernhard Egger, Marcel Luthi, Sandro Sch¨onborn, and
Thomas Vetter. Morphable face models-an open framework.
In 2018 13th IEEE international conference on automatic
face & gesture recognition (FG 2018), pages 75–82. IEEE,
2018. 2
[9] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial networks. Commu-
nications of the ACM, 63(11):139–144, 2020. 1
[10] Philip-William Grassal,
Malte Prinzler,
Titus Leistner,
Carsten Rother, Matthias Nießner, and Justus Thies. Neural
head avatars from monocular rgb videos. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18653–18664, 2022. 2
[11] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh recon-
struction and high-quality mesh rendering. arXiv preprint
arXiv:2311.12775, 2023. 3
[12] Antoine Gu´edon and Vincent Lepetit.
Gaussian frosting:
Editable complex radiance fields with real-time rendering.
arXiv preprint arXiv:2403.14554, 2024. 3
[13] Abdullah Hamdi, Luke Melas-Kyriazi, Guocheng Qian, Jin-
jie Mai, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and
Andrea Vedaldi. Ges: Generalized exponential splatting for
efficient radiance field rendering. arXiv, 2024. 3
[14] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. SIGGRAPH, 2024. 3
[15] Berna Kabadayi, Wojciech Zielonka, Bharat Lal Bhatnagar,
Gerard Pons-Moll, and Justus Thies. Gan-avatar: Control-
lable personalized gan-based human head avatar. In Interna-
tional Conference on 3D Vision (3DV), 2024. 1
[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(TOG), 42(4):1–14, 2023. 1, 3
[17] Taras Khakhulin, Vanessa Sklyarova, Victor Lempitsky, and
Egor Zakharov. Realistic one-shot mesh-based head avatars.
In European Conference on Computer Vision, pages 345–
362. Springer, 2022. 2
[18] Diederik P Kingma and Max Welling. Auto-encoding varia-
tional bayes. arXiv preprint arXiv:1312.6114, 2013. 1
[19] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim
Walter, and Matthias Nießner. Nersemble: Multi-view ra-
diance field reconstruction of human heads.
ACM Trans.
Graph., 42(4), 2023. 1
[20] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian representation for
radiance field. arXiv preprint arXiv:2311.13681, 2023. 3
[21] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier
Romero. Learning a model of facial shape and expression
from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017. 2
[22] Ancheng Lin and Jun Li.
Direct learning of mesh and
appearance via 3d gaussian splatting.
arXiv preprint
arXiv:2405.06945, 2024. 3
9

<!-- page 10 -->
[23] Stephen Lombardi, Jason Saragih, Tomas Simon, and Yaser
Sheikh. Deep appearance models for face rendering. ACM
Transactions on Graphics (ToG), 37(4):1–13, 2018. 2
[24] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural vol-
umes: learning dynamic renderable volumes from images.
ACM Transactions on Graphics (TOG), 38(4):1–14, 2019. 2
[25] Stephen Lombardi,
Tomas Simon,
Gabriel Schwartz,
Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mix-
ture of volumetric primitives for efficient neural rendering.
ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 2
[26] Shugao Ma, Tomas Simon, Jason Saragih, Dawei Wang,
Yuecheng Li, Fernando De La Torre, and Yaser Sheikh. Pixel
codec avatars. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 64–73,
2021. 2, 4, 6, 7
[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
1
[28] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and
Peter Eisert.
Compact 3d scene representation via self-
organizing gaussian grids. arXiv preprint arXiv:2312.13299,
2023. 3
[29] Frederick I Parke. Computer generated animation of faces. In
Proceedings of the ACM annual conference-Volume 1, pages
451–457, 1972. 1
[30] Stanislav Pidhorskyi, Tomas Simon, Gabriel Schwartz, He
Wen, Yaser Sheikh, and Jason Saragih.
Rasterized edge
gradients: Handling discontinuities differentiably.
arXiv
preprint arXiv:2405.02508, 2024. 4
[31] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide
Davoli, Simon Giebenhain, and Matthias Nießner.
Gaus-
sianavatars: Photorealistic head avatars with rigged 3d gaus-
sians. arXiv preprint arXiv:2312.02069, 2023. 3
[32] Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan
Li, and Giljoo Nam. Relightable gaussian codec avatars. In
CVPR, 2024. 1, 3, 4, 6
[33] Kripasindhu Sarkar, Marcel C B¨uhler, Gengyan Li, Daoye
Wang, Delio Vicini, J´er´emy Riviere, Yinda Zhang, Sergio
Orts-Escolano, Paulo Gotardo, Thabo Beeler, et al. Litnerf:
Intrinsic radiance decomposition for high-quality view syn-
thesis and relighting of faces. In SIGGRAPH Asia 2023 Con-
ference Papers, pages 1–11, 2023. 1
[34] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang,
Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang.
SplattingAvatar: Realistic Real-Time Human Avatars with
Mesh-Embedded Gaussian Splatting.
In Computer Vision
and Pattern Recognition (CVPR), 2024. 3
[35] Jingxiang Sun, Xuan Wang, Lizhen Wang, Xiaoyu Li, Yong
Zhang, Hongwen Zhang, and Yebin Liu. Next3d: Gener-
ative neural texture rasterization for 3d-aware head avatars.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20991–21002, 2023. 2
[36] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio
Del Bue. Haha: Highly articulated gaussian human avatars
with textured mesh prior. arXiv preprint arXiv:2404.01053,
2024. 3, 7
[37] Joanna Waczy´nska, Piotr Borycki, Sławomir Tadeja, Jacek
Tabor, and Przemysław Spurek. Games: Mesh-based adapt-
ing and modification of gaussian splatting. arXiv preprint,
2024. 3
[38] Cong Wang, Di Kang, He-Yi Sun, Shen-Han Qian, Zi-Xuan
Wang, Linchao Bao, and Song-Hai Zhang. Mega: Hybrid
mesh-gaussian head avatar for high-fidelity rendering and
head editing.
arXiv preprint arXiv:2404.19026, 2024.
3,
7
[39] Daoye Wang, Prashanth Chandran, Gaspard Zoss, Derek
Bradley, and Paulo Gotardo.
Morf: Morphable radiance
fields for multiview neural head modeling.
In ACM SIG-
GRAPH 2022 Conference Proceedings, pages 1–9, 2022. 1
[40] Cheng-hsin Wuu, Ningyuan Zheng, Scott Ardisson, Rohan
Bali, Danielle Belko, Eric Brockmeyer, Lucas Evans, Tim-
othy Godisart, Hyowon Ha, Xuhua Huang, Alexander Hy-
pes, Taylor Koska, Steven Krenn, Stephen Lombardi, Xi-
aomin Luo, Kevyn McPhail, Laura Millerschoen, Michal
Perdoch, Mark Pitts, Alexander Richard, Jason Saragih,
Junko Saragih, Takaaki Shiratori, Tomas Simon, Matt Stew-
art, Autumn Trimble, Xinshuo Weng, David Whitewolf,
Chenglei Wu, Shoou-I Yu, and Yaser Sheikh. Multiface: A
dataset for neural face rendering. In arXiv, 2022. 1
[41] Jun Xiang, Xuan Gao, Yudong Guo, and Juyong Zhang.
Flashavatar: High-fidelity head avatar with efficient gaussian
embedding. In The IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2024. 3
[42] Yuting Xiao, Xuan Wang, Jiafei Li, Hongrui Cai, Yanbo Fan,
Nan Xue, Minghui Yang, Yujun Shen, and Shenghua Gao.
Bridging 3d gaussian and mesh for freeview video rendering.
arXiv preprint arXiv:2403.11453, 2024. 3
[43] Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang,
Lizhen Wang, Zerong Zheng, and Yebin Liu. Gaussian head
avatar: Ultra high-fidelity head avatar via dynamic gaussians.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024. 3
[44] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 7
[45] Zhongyuan Zhao, Zhenyu Bao, Qing Li, Guoping Qiu, and
Kanglin Liu.
Psavatar: A point-based morphable shape
model for real-time head avatar creation with 3d gaussian
splatting. arXiv preprint arXiv:2401.12900, 2024. 3
[46] Yufeng Zheng, Victoria Fern´andez Abrevaya, Marcel C.
B¨uhler, Xu Chen, Michael J. Black, and Otmar Hilliges. I
M Avatar: Implicit morphable head avatars from videos. In
Computer Vision and Pattern Recognition (CVPR), 2022. 2
[47] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Instant
volumetric head avatars. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 4574–4584, 2023. 2
10
