<!-- page 1 -->
1
POTR: Post-Training 3DGS Compression
Bert Ramlot
, Martijn Courteaux
, Peter Lambert
, Senior Member, IEEE,
Glenn Van Wallendael
, Member, IEEE
Abstract—3D Gaussian Splatting (3DGS) has recently emerged
as a promising contender to Neural Radiance Fields (NeRF)
in 3D scene reconstruction and real-time novel view synthesis.
3DGS outperforms NeRF in training and inference speed but
has substantially higher storage requirements. To remedy this
downside, we propose POTR, a post-training 3DGS codec built
on two novel techniques. First, POTR introduces a novel pruning
approach that uses a modified 3DGS rasterizer to efficiently
calculate every splat’s individual removal effect simultaneously.
This technique results in 2-4× fewer splats than other post-
training pruning techniques and as a result also significantly ac-
celerates inference with experiments demonstrating 1.5-2× faster
inference than other compressed models. Second, we propose
a novel method to recompute lighting coefficients, significantly
reducing their entropy without using any form of training. Our
fast and highly parallel approach especially increases AC lighting
coefficient sparsity, with experiments demonstrating increases
from 70% to 97%, with minimal loss in quality. Finally, we extend
POTR with a simple fine-tuning scheme to further enhance prun-
ing, inference, and rate-distortion performance. Experiments
demonstrate that POTR, even without fine-tuning, consistently
outperforms all other post-training compression techniques in
both rate-distortion performance and inference speed.
Index Terms—3DGS, compression, spherical harmonics, en-
ergy compaction, pruning.
I. INTRODUCTION
Synthesizing new views from a limited number of camera-
captured images is a long-standing problem in computer
graphics [1]–[3]. Various techniques have been proposed to
address this challenge, focusing on multiview video cod-
ing [4], [5], cross-view image matching [6], sparse compact
representations [7], and real-time VR experiences [8], [9].
Among the most promising solutions are Neural Radiance
Fields (NeRF) [10] and 3D Gaussian Splatting (3DGS) [11].
While NeRF has seen substantial year-over-year improvements
[12]–[14], 3DGS achieves better training and inference speeds
through a scene representation based on feature-rich volumet-
ric points called splats. This alternate representation is also
the culprit behind 3DGS’s biggest comparative downside to
NeRF, namely, substantially higher storage requirements. For
example, a simple unbounded scene with a central object
typically results in a model size of 0.3-1.5 GB, often making
This work was funded in part by the Research Foundation—Flanders
(FWO) under Grant 1SA0B26N, the imec.prospect project SitSens, ID-
Lab (Ghent University—imec), Flanders Innovation and Entrepreneurship
(VLAIO), and the European Union.
The authors are with the IDLab-MEDIA research group, part of Ghent
University and imec, located at AA Tower, Technologiepark-Zwijnaarde 122,
B-9052 Zwijnaarde, Belgium. Corresponding author: Bert Ramlot. (e-mail:
bert.ramlot@ugent.be; martijn.courteaux@ugent.be; peter.lambert@ugent.be;
glenn.vanwallendael@ugent.be).
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.
transfer and storage challenging, particularly for on-demand
applications. Furthermore, with 3DGS literature expanding
toward larger scenes [15], [16] and immersive video [17], [18],
3DGS compression is becoming increasingly important.
The literature has recognized the need for compres-
sion through numerous 3DGS-specific compression tech-
niques [19]–[51], most of whom are in-training compression
techniques [19]–[44] that alter the training process to achieve
smaller models. While this works well, this level of control
over the training process is not guaranteed as one might
want to re-encode an existing 3DGS model, similarly to how
videos and images are re-encoded. To this end, a fraction
of the 3DGS compression literature focuses on post-training
compression [45]–[51] as they start from an existing model.
Lack of control over the training process generally results in
poorer rate-distortion (RD) performance. As a result, post-
training compression approaches commonly incorporate a fine-
tuning step to somewhat level the playing field. This step
further trains the already trained model to recover some of the
quality lost during compression. Although this is a powerful
way for a post-training codec to utilize the extra flexibility
offered by the in-training compression paradigm, it side-steps
the issue of post-training compression. Nevertheless, current
post-training methods primarily focus on RD performance
after fine-tuning.
To this end, we propose POTR, a post-training 3DGS
codec that focuses predominantly on achieving strong RD
performance without any fine-tuning. POTR significantly out-
performs existing post-training methods, achieving up to a
fourfold reduction in model size compared to the previous
state-of-the-art. Additionally, our compressed models use far
fewer splats and therefore render faster. This markedly bet-
ter performance is achieved through two novel compression
techniques.
Our first novel technique introduces a new approach to
splat removal. Where existing post-training pruning methods
rely on heuristics involving metrics such as a splat’s size,
opacity, and importance, we propose a pruning technique that
directly evaluates the impact of removing each splat on an
objective quality metric. This approach allows our high-quality
compressed models to use 2–4× fewer splats than other post-
training methods across four common datasets. Additionally,
our pruning method significantly accelerates inference, often
achieving at least 50% higher frames per second. At higher
distortion levels, the performance gap widens further, with
experiments showing examples of 2× higher frame rates than
other post-training compressed models.
Our second novel technique focuses on spherical harmonics
coefficients — commonly referred to as lighting coefficients
— which represent a splat’s view-dependent color. Lighting
arXiv:2601.14821v1  [cs.CV]  21 Jan 2026

<!-- page 2 -->
2
coefficients constitute over 80% of the uncompressed model’s
size, making them vital to compression. To address this, we
first represent a splat’s colors for all training views using a
single linear system. Next, we introduce a spherical harmon-
ics energy compaction method that uses a heavily modified
version of ridge regression to compute an alternate set of
lighting coefficients. These new coefficients exhibit signifi-
cantly lower entropy while producing nearly identical colors
for relevant training views and generalizing better to novel
views. Combined with quantization and entropy compression,
we demonstrate that lighting coefficients are no longer the
largest contributor to model size using our spherical harmonics
energy compaction method. To our knowledge, this is the
first post-training 3DGS compression method to non-trivially
recompute lighting coefficients without any form of training.
In summary, the main contributions of this paper are:
• Proposing an efficient method to evaluate the impact
of removing each splat on an objective quality metric.
Use this to design a pruning strategy that significantly
outperforms other post-training methods in both RD
performance and inference speed, particularly at higher
distortion levels.
• Introducing a fast, systematic approach to transform high-
entropy spherical harmonics coefficients into low-entropy
ones while preserving the splat’s colors for relevant
training views. Our method requires no training, is em-
barrassingly parallel, and improves generalization.
• Developing POTR, a fine-tuneless post-training codec
that leverages the above two techniques to achieve state-
of-the-art RD performance and inference speed, surpass-
ing all other (fine-tuning-based) post-training methods.
Additionally, we demonstrate that extending POTR with
a simple fine-tuning scheme further enhances RD perfor-
mance and inference speed.
The remainder of this work is organized as follows. Sec-
tion II briefly overviews the relevant literature. Next, Sec-
tion III discusses the necessary background on 3DGS which
will be used extensively in Section IV which provides a
detailed description of our proposed compression methods
and codec. Section V presents our results, experiments, and
associated discussions. Finally, Section VI summarizes this
work.
II. RELATED WORK
A. NeRF compression
Voxel-based techniques are a popular solution for enhanc-
ing the training and rendering speeds of NeRFs [52], [53].
However, these methods often lead to large storage over-
head, for example, KiloNeRF [53] necessitates the storage of
thousands of neural networks. As a result, a large fraction
of the NeRF compression literature focuses on voxel-based
techniques. Voxel-based techniques are comparable to 3DGS
in that their representation is more localized. This localization
generally eases compression as it improves existing, or allows
for new, compression techniques such as voxel pruning [54]–
[56], transform coding [56]–[61], various forms of quanti-
zation [55], [62], [63], and specialized context models for
entropy compression [64].
B. Point-cloud compression
Point-cloud geometry is commonly compressed using oc-
trees [65]–[67]. The octree structure is serialized by en-
coding the occupancy bits of the octree’s nodes, which are
subsequently compressed using entropy coding. To achieve
higher compression ratios, several custom entropy models have
been proposed. Examples include a tree-structured entropy
model utilizing MLPs [67], an intra-prediction-based entropy
model [65], and a deep entropy model [66].
The compression of point-cloud attributes, such as color
values and normal vectors, is another key area of research [68].
While fully 3D methods often use octrees to encode at-
tributes [69]–[71], alternative approaches where 3D data is
mapped onto 2D planes, allowing the use of standard image
and video compression techniques [72], do occur.
C. 3D Gaussian Splatting compression
To address 3DGS’s stringent storage requirements, several
compression techniques have been proposed [19]–[51]. Early
approaches are comprehensively surveyed in 3DGS.zip [73],
which also introduces a methodology to compare different
3DGS compression techniques. Notably, the 3DGS literature
bears close resemblance to point-cloud compression liter-
ature by, for example, compressing splat geometry using
octrees [28], [46] and projecting 3D attributes onto 2D
planes to allow for image-based compression [20]–[22]. Some
works even directly use point cloud codecs, for example, G-
PCC [23], [24], [26], [43], [50]. While point-cloud-inspired
techniques result in substantial RD gains, the largest gains of
the 3DGS literature originate from 3DGS-specific techniques.
A first group of 3DGS-specific compression techniques
focuses on reducing the number of splats. Most approaches
assign each splat a heuristic score, computed using factors
such as sensitivity [47], opacity [22]–[26], [30], [44], [45],
importance [29], [38], [41], [46], intersection [37], [41],
maximum contribution [38], [40], uncertainty [39], [49] or
volume [45], [46], [50]. Splats with scores below a chosen
threshold are then removed [22]–[26], [29], [30], [33], [45]–
[47]. Another more indirect approach is to apply a downward
pressure on the number of splats during training by altering
the loss function [19], [25]–[28], [31], [32], [34], [35], [40],
[42]. Both methods have been quite successful at reducing the
number of splats, so much so that the largest inference time
improvements have thus far originated from the compression
literature, highlighting its contribution to Gaussian Splatting
beyond size reduction.
A second group of 3DGS-specific compression techniques
addresses the substantial storage requirements of lighting
information. In 3DGS’s original formulation, lighting infor-
mation is modeled using spherical harmonics (SH) and their
coefficients. The latter are stored and account for over 80% of
all attributes, leading some works to avoid explicitly saving
lighting coefficients altogether. Instead, latent attributes are
stored and processed using an MLP to generate either direct

<!-- page 3 -->
3
color outputs [23]–[26], [30], [31], [34], [36], [42], or lighting
coefficients [20], [21], [29], [32]. For methods that store
the lighting coefficients explicitly, additional techniques are
applied to lower the lighting coefficients’ entropy. Examples
include trainable masks [28], [35], SH band pruning after
densification [33], distillation of higher-degree SH terms to
lower degrees [45], and transformations such as Region-
Adaptive Hierarchical Transform [37], [46] and JPEG XL [22].
To summarize, a range of techniques are used to compress
3DGS models, but those that focus on reducing the number of
splats or compressing lighting information are the most crucial
to achieving strong RD performance. Compressing other prop-
erties — such as position, scale, rotation, and opacity — is
comparatively less important, as these contribute only a small
portion to the overall uncompressed size. Moreover, reducing
the number of splats not only enhances RD performance but
also boosts inference speed, underscoring the importance of
compression techniques beyond just storage efficiency.
III. 3D GAUSSIAN SPLATTING
3DGS models consist of a collection of splats, each with
a center µ ∈R3 in space. For a point in space p ∈R3,
the splat’s density is proportional to a 3D Gaussian with a
covariance matrix Σ ∈R3×3:
G(p) = exp (−1
2(p −µ)T Σ−1(p −µ)) .
(1)
To render a 3DGS model for a given camera, a splat’s
density is first projected into screen space. This is done
efficiently using the affine invariance of multivariate Gaussians
by approximating the projective transformation as an affine
transformation. The resulting 2D density is therefore again
proportional to a 2D Gaussian and denoted as G2D. Next, the
splats are alpha-composited by associating each splat with
a layer and ordering these layers according to the splats’
approximate depth. Upon rasterizing a splat’s layer, a pixel
at position x ∈R2 has an opacity
α = o · G2D(x) ,
(2)
where o represents the location-independent base opacity of
the splat. The yet undetermined fraction of a pixel’s color after
alpha compositing the first k layers is termed the transmittance
and is defined as
Tk =
k−1
Y
j=0
(1 −αj) .
(3)
After alpha-compositing all K visible splats over a black
background, the final color of a pixel located at x ∈R2 is
given by
c(x) =
K−1
X
k=0
Tkαkck
(4)
where ck is the color of the k-th splat in the alpha compositing
process.
While a splat’s color is monochromatic across the pixels
of a still frame, it is view-dependent based on the cam-
era’s position through a spherical function defined using real
spherical harmonics. 3DGS’s original formulation uses all real
0.5
0.0
0.5
Y1
Y2
Y3
Y4
Y5
Y6
Y7
Y8
Y9
Y10
Y11
Y12
Y13
Y14
Y15
Y16
Fig. 1: Mollweide projection of the first 16 real spherical
harmonics.
spherical harmonics with a degree l ≤3, all of whom are
shown in Figure 1. To ease iterating these functions, each
degree-order pair (l, m) is mapped to a unique index using
(l, m) →l(l + 1) + m + 1. The first spherical harmonic Y1 is
often singled out and called the DC spherical harmonic as it is
the only basis that is view-independent and the only basis that
is not zero-meaned. All other spherical harmonics are called
AC spherical harmonics. The value of a splat’s color channel
is determined by the M lighting coefficients associated with
that splat and channel. It is evaluated based on the direction
d extending from the camera to the splat center, as follows:
C(d) =
M
X
i=1
LiYi(d),
(5)
where Li represents the lighting coefficient corresponding to
Yi.
IV. METHOD
This section discusses the different steps of our proposed
encoder, for which Figure 2 provides an overview. First,
Section IV-A introduces a preliminary metric that is used
throughout POTR. Next, we focus on splat removal, the
first step of our encoder, with Section IV-B detailing how
our proposed method can efficiently and accurately calculate
the effect of a splat’s removal. This information is used in
Section IV-C as the basis for our proposed splat-removal
method. After pruning, we use a novel spherical harmonics
energy compaction method to reduce the lighting coefficients’
entropy in Section IV-D. To complete our codec, Section IV-E
discusses tangential topics such as quantization, serialization,
entropy compression, and how these are all combined. Finally,
Section IV-F discusses how our codec can be adapted to
incorporate fine-tuning.
A. Splat importance
POTR utilizes lossy compression, during which a splat may
be removed or its properties altered. The impact of modifying
or removing a splat on the quality of the model varies heavily
based on the splat in question. To help upcoming lossy steps
gauge this impact, we define the importance Ik(s) of the k-th
splat to a camera s as the fraction of the camera’s synthesized
image that originates from the k-th splat, i.e.
Ik(s) = 1
P
X
x
Tikαik
(6)

<!-- page 4 -->
4
010100
0101 0001 1101
Reorder splats
...
scale rot
sh
splat order
Uniform
quantization
Octree
quantization
zstd
...
...
...
BIN
(c) Quantization
(d) Entropy compression
old SH
coef.
new SH
coef.
other properties
(b) SH energy compaction
...
Per splat (parallel)
L1
L2
L3
L4
0.5
2.9
0.2
0.4
0.6
0.0
0.1
0.0
Before
After
PLY
No
No
Remove
Yes
Yes
attributes
geometry
(a) Pruning
△MSE(    )
△MSE(    )
△MSE(    )
≤ △MSEMAX
Continue
pruning?
Calculate
 removal cost
Fig. 2: Simplified overview of POTR. (a) Splats are removed across multiple pruning iterations based on the change in the
model’s mean square error (MSE) upon their removal. (b) Spherical harmonic coefficients are energy compacted, yielding a
new set of lighting coefficients with a lower entropy. (c) Splat geometry is quantized and serialized using an octree, then
attributes are uniformly quantized and serialized using the spatial order implied by the depth-first traversal of the octree. (d)
The serialized data is entropy compressed using zstd, resulting in the final compressed bitstream.
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
Fraction of highest importance splats
Normalized
cumulative importance
Fig. 3:
Normalized cumulative importance of the highest
importance splats in the Truck model of the initial 3DGS
implementation [11]. The cumulative importance is the sum
of the per-splat importances Ik for a given subset of splats.
where P is the number of pixels and ik is the rank of the
k-th splat in the alpha-compositing process for the pixel x.
The importance of a splat can be generalized to account for
the entire scene, encompassing all N cameras, as follows:
Ik = 1
N
X
s
Ik(s) .
(7)
Figure 3 shows that a small fraction of splats draws the vast
majority of a scene, highlighting the immense potential of
selective loss introduction in a compression context.
B. Effect of a splat’s removal
Removing splats is a common way to reduce the final
file size. To help determine if a splat is to be kept or
removed, we desire a score that describes the impact of a
splat’s removal on the quality of the scene. Other post-training
compression methods rely solely on heuristics to set this score,
for example, using a splat’s size or importance Ik. However,
these approaches tend to be sub-optimal as they overlook
crucial factors such as color information and the intricate
geometry of the model. For instance, the effect of removing
a red splat varies depending on the splats that lie behind it.
If this is another red splat, the change in the rendered image
will be relatively minor. Conversely, the visual difference will
be much more pronounced if a blue splat lies behind the red
splat. Metrics such as a splat’s importance Ik fail to capture
such nuances.
To address this limitation, we propose to modify the original
rasterizer’s forward rendering pass to accurately evaluate the
effect of removing a splat, for all splats simultaneously. Our
proposed modification utilizes readily available data, specif-
ically the opacities of the contributing splats for each pixel
and the partial colors {Pi}. The latter is a pixel’s color after i
splats have been considered in the alpha compositing process
and can be expressed as
Pi =
i−1
X
k=0
Tkαkck .
(8)
Removing the k-th splat with rank ik in the alpha-compositing
process of a given pixel renormalizes the transmittance of all
subsequent splats by a factor of
1
1−αik . Using this observation
together with Equations (4) and (8), we derive a compact ex-
pression for the pruning difference PDk(x), which quantifies
the change in a pixel’s color resulting from the removal of the
k-th splat:
PDk(x) = ˜ck(x) −c(x)
= (
ik−1
X
j=0
Tjαjcj +
1
1 −αik
K−1
X
j=ik+1
Tjαjcj) −c(x)
= (Pik +
1
1 −αik
(PK −Pik+1)) −PK
= Pik −
1
1 −αik
Pik+1 +
αik
1 −αik
PK ,
(9)

<!-- page 5 -->
5
−1
−0.5
0.5
1
0.5
1
x
m(x)
Fig. 4: The mapping function m(x) for a = 10.
where c(x) and ˜ck(x) denote a pixel’s color before and after
removing the k-th splat, respectively.
The pruning difference is crucial for determining the impact
of each splat’s removal on an objective quality metric. In this
initial work, the squared error is used due to its simplicity
and relation to the PSNR. The difference in squared error, per
pixel and color channel, as a result of removing the k-th splat
is given by
△SEk(x) = (˜ck(x) −ct(x))2 −(c(x) −ct(x))2
= (˜ck(x) −c(x))(˜ck(x) + c(x) −2ct(x))
= (PDk(x))2 + 2 PDk(x) (c(x) −ct(x)) ,
(10)
where ct(x) is a fixed reference target color, defined as the
initial synthesized color of the uncompressed model1 such that
initially ct(x) = c(x). In Section IV-C, we will recompute
△SEk(x) after removing splats, which changes c(x), ˜ck(x),
and consequently PDk(x), but not ct(x). By averaging the
change in squared error across all N cameras, P pixels, and
3 color channels, we arrive at the overall effect of the k-th
splat’s removal on the mean squared error of the model:
△MSEk = 1
N
X
s
1
P
X
x
1
3△SEk(x) · 1 .
(11)
Finally, it should be noted that our CUDA implementation
of the above method is deliberately designed to be interop-
erable with the standard forward-backward render pass used
during training. The forward pass comes at no extra cost if
our altered render pass is already executed to calculate the
splats’ removal effects. Future work could explore how our
implementation can be properly integrated into the training
process.
C. Iterative pruning
The pruning controller manages the removal of splats based
on the △MSEk values. It operates under the assumption that
the cumulative effect of removing a group of splats S is
approximately the sum of the individual effects, i.e.
△MSES ≈
X
k∈S
△MSEk .
(12)
This assumption, which we call the pruning approximation,
tends to grossly underestimate △MSES when removing too
1Another valid choice for the target color ct(x) is the ground truth color.
However, our limited experiments indicate that its usage does not necessarily
yield better results. Furthermore, since this work focuses on post-training
compression, ground truth images are not always available.
many splats at once. To counteract this limitation, we propose
an iterative approach with multiple pruning iterations where all
△MSEk values are recalculated in between pruning iterations.
In each iteration, the set of splats S to be removed is
constructed by iteratively adding splats to an empty set until
the pruning budget Bprune is reached. We use the importance of
a splat Ik as a simple estimate for the splat’s ability to distort
the pruning approximation, defining the pruning budget to be
reached when
Bprune ≤
X
k∈S
Ik .
(13)
The pruning controller aims to remove all splats where
△MSEk < △MSEMAX, and adjusts its pruning budget accord-
ingly every iteration to achieve this:
Bprune =
1
# remaining iterations
X
k∈{ j | △MSEj<△MSEMAX }
Ik .
(14)
The pruning controller adds splats to S in ascending order of
their m( △MSEk
△MSEMAX ) value where
m(x) =
(
x
x ≥0
1
a(√1 −2a · x −1)
x < 0
(15)
is a mapping function (see Figure 4). This function is designed
to prioritize the removal of splats with a small or negative
△MSEk value, with the parameter a controlling the balance
between these two objectives. By prioritizing splats with small
△MSEk, often more than 50% of the splats can be removed
in just one iteration, accelerating subsequent pruning. On the
other hand, focusing on splats with a negative △MSEk value
helps minimize quality loss during the pruning process.
D. Spherical harmonics energy compaction
Spherical harmonics, and their coefficients, define a splat’s
color for every direction. However, in practice, most splats
are sampled only from a small subset of directions. This
phenomenon arises for various reasons, including occlusion
between splats, and the observer’s restricted range of motion.
This section’s key idea is to use this phenomenon to find
alternate, easier-to-compress, lighting coefficients that retain
the color within the sampled subset while being agnostic
to color changes outside the sampled subset. Since color
changes only occur for unsampled directions, the model’s
overall quality is unaffected.
To find an alternate set of lighting coefficients, we first
observe that Equation (5) holds for all N training cameras
resulting in N equations per splat that can be rewritten as
follows:



C(d1)
C(d2)
...
C(dN)



|
{z
}
C
=



Y1(d1) Y2(d1) ···
YM(d1)
Y1(d2) Y2(d2) ···
YM(d2)
...
...
...
...
Y1(dN) Y2(dN) ··· YM(dN)



|
{z
}
Y
×


L1
L2
...
LM


| {z }
L
(16)
We propose to use conventional energy compaction to find
a new L, denoted as L′, located on the RD Pareto front by

<!-- page 6 -->
6
using all N training cameras. This work uses ridge regression
to define the optimal trade-off:
L′ = arg min
x∈RM×1(∥Yx −C∥2
2 + ∥Γx∥2
2)
(17)
where ∥·∥2 is the Euclidean norm of a vector, and Γ de-
notes the Tikhonov matrix which defines how the lighting
coefficients are regularized. The Tikhonov matrix is chosen
as follows:
Γ =
√
λ

0
01×M−1
0M−1×1
IM−1×M−1

(18)
where λ is the regularization coefficient and I denotes the
identity matrix. The top-left element is zero as we do not
wish to regularize the DC lighting coefficient. The benefit of
ridge regression, and the reason it was chosen for this work as
opposed to other regularization techniques, is its closed-form
solution:
L′ = (YT Y + ΓT Γ)−1YT C
(19)
A closed-form solution is essential as L′ has to be calculated
for each splat, with models having hundreds of thousands to
millions of splats.
With the core idea in place, we propose four additional
modifications to improve RD performance further:
1) Importance: Equation (17) punishes each deviation from
C equally, regardless of the importance of the splat to the
camera. However, we expect a deviation in a splat’s color (as
perceived by a given camera) to affect the quality of a model
approximately in accordance with the importance of that splat
to that camera. We can take this into account by doing the
following substitutions in Equation (17) (and Equation (19)):
C ←C ⊙col (Ik(s1), ..., Ik(sN))T
(20)
Y ←Y ⊙col (Ik(s1), ..., Ik(sN))T
(21)
where ⊙col represents a column element-wise multiplication.
2) Color
model:
RGB
lighting
coefficients
correlate
highly across color channels. By switching to a luminance-
chrominance color model, we decorrelate the color chan-
nels and as a result, reduce the total energy. Additionally,
chrominance channels can be more aggressively regularized,
as human vision is less sensitive to chrominance changes. For
this work, we use the YCoCg color model and regularize the
chrominance channels thrice as hard by setting λ = λY =
1
3λCo =
1
3λCg. Surprisingly, to the best of our knowledge,
this is the first post-training 3DGS compression work to use
a non-RGB color model.
3) Sparsity: Due to the limited set of sampled directions,
column vectors of Y often exhibit quasi-parallelism among
themselves, suggesting that lighting coefficients can be spar-
sified. Unfortunately, ridge regression spreads energy across
parallel column vectors, leading to a non-sparse and under-
regularization solution. We address this by zeroing out the i-th
column of Y, which leads to L′
i = 0, if a preceding column
is ’sufficiently parallel’ to it. Two column vectors are said to
be sufficiently parallel if the absolute value of their cosine
similarity is larger than some threshold α.
4) Two-pass regularization: Ridge regression forces a large
number of AC lighting coefficients to be almost zero such
that these lighting coefficients will be quantized to zero in
Section IV-E. To reduce the quantization loss and to present
ridge regression with a more true representation of its flexibil-
ity, a two-pass regularization system is used. The first pass
identifies all almost zero AC coefficients and subsequently
forces them to zero. The second pass finds the optimal values
for the remaining non-zero lighting coefficients.
In summary, we demonstrate that a splat’s color for all N
training cameras can be expressed as one linear system of
lighting coefficients. We then propose to use ridge regres-
sion to find a new set of lighting coefficients with a lower
entropy. Finally, we improve upon this basic energy com-
paction method through importance weighting, a luminance-
chrominance model, the removal of parallel column vectors,
and a two-pass regularization system.
E. Codec
We use the proposed pruning and spherical harmonics
energy compaction methods to create a codec (see Figure 2).
The encoder first removes redundant splats before spherical
harmonics energy compaction reduces the entropy of the
remaining splats’ lighting coefficients. Subsequently, the ge-
ometry and attributes are quantized and serialized contiguously
into a bitstream, iterating over all splats before serializing the
next property. Finally, the bitstream is compressed using zstd2,
an out-of-the-box lossless compression algorithm, yielding the
final compressed bitstream. The decoder follows the opposite
process by decompressing using zstd, deserializing, and de-
quantizing the bitstream, in that order.
Quantization is performed using multiple strategies. The
splat attributes (scale, rotation, opacity, and SH coef.) are uni-
formly quantized and dequantized using the transformations
ˆx =
1
2 + x · SF

and
˜x = ˆx
SF ,
(22)
where SF is the scale factor, a hyperparameter tailored to
each attribute. Unlike attribute quantization, the geometry of
the splats is quantized using an octree structure. Each splat’s
position µ is quantized to the center of an octree leaf ˆµ. The
octree is constructed by repeatedly splitting leaves containing
multiple splats or failing to meet the precision criterion
∥µ −ˆµ∥2 < max

γ, β · min
s ∥µ −µs∥2

,
(23)
where ∥·∥2 denotes the Euclidean distance, µs is the position
of the eye of camera s, and β and γ are hyperparameters.
This criterion leverages the observation that positional changes
of distant splats have minimal impact on their screen-space
positions. We serialize the octree using a depth-first traversal.
Upon visiting a node for the first time, we append its occu-
pancy to the bitstream. The occupancy of an octree’s node
is a single byte in which each bit corresponds to one of the
eight octants and indicates whether that octant contains a child
node.
2https://github.com/facebook/zstd

<!-- page 7 -->
7
Uncompressed model
Encode model
(using POTR)
RFT > 0
Compressed model
Decode model
(using POTR)
Fine-tune
RFT ←RFT −1
Yes
No
Fig. 5: Schematic representation of the POTR-FT encoder.
RFT denotes the remaining number of fine-tuning cycles.
Our codec also employs some additional optimizations. For
instance, DC lighting coefficients are differentially encoded,
the real
quaternion component of the rotation is omitted,
and opacity reconstruction values are shifted off-center during
dequantization by using ˜x = ˆx+0.25
SF
(see Equation (22)). We
also note that the order of the steps shown in Figure 2 is
flexible. For example, spherical harmonics energy compaction
can be applied earlier, e.g., after half of the pruning iterations,
to allow the remaining iterations to account for the changes
introduced by energy compaction.
F. Fine-tuning
To enable a fair comparison of our proposed codec with
others that incorporate fine-tuning, we introduce a variant
of POTR, called POTR-FT, which includes a fine-tuning
mechanism. Since fine-tuning is not the focus of this work,
we deliberately keep POTR-FT’s design simple. Specifically,
its encoder employs the original (fine-tuneless) encoder and
decoder as an internal black-box component, performing RFT
cycles of decode →fine-tune →encode, following the initial
encoding. The fine-tuning step trains the model using the
original 3DGS training routine. The encoder design of POTR-
FT is depicted in Figure 5.
V. EXPERIMENTS AND DISCUSSION
We present and discuss our codec’s performance using the
methodology laid out by 3DGS.zip [73], a 3DGS compression
survey focusing on low distortion. If 3DGS.zip’s methodology
provides no guidance, we follow the initial 3DGS implemen-
tation’s [11] approach where possible.
The remainder of this section is structured as follows.
First, we discuss our experimental settings in Section V-A.
Afterward, we present POTR’s quantitative and qualitative
results in Section V-B. In Section V-C and Section V-D we take
a closer look at our proposed pruning and spherical harmonics
energy compaction approach respectively. In Section V-E we
present two ablation studies, and in Section V-F we discuss
our codec’s speed.
A. Experimental settings
1) Datasets: We evaluate our codec across four datasets:
three COLMAP [74] datasets (Mip-NeRF 360 [13], Deep
Blending [75], and Tanks And Temples [76]) and one Blender
dataset (NeRF-Synthetic [10]). For the COLMAP datasets,
every 8th image is designated for testing, while the Blender
dataset comes with a predefined train-test split. The additional
Mip-NeRF 360 scenes (flowers and treehill) are included and
outdoor scenes (bicycle, flowers, stump, treehill, garden) are
downscaled 4× while indoor scenes (counter, kitchen, room,
bonsai) are downscaled 2x.
2) Models: As POTR is a post-training compression codec,
a trained model is required to use our proposed codec.
We use the models published alongside the initial 3DGS
implementation3 where possible, however, NeRF-Synthetic
models are absent from this collection. We train these models
ourselves using the configuration provided by 3DGS’s initial
implementation.
3) Hyperparameters: All hyperparameters are governed by
a single quality parameter q as specified by Table II. We set
q to 0.5 in spirit, but to match the results of other works,
we alter it slightly per dataset such that POTR’s objective
quality slightly exceeds those of other compression works. In
comparison, MesonGS [46] sets their hyperparameters on a
per-scene basis. This yields further non-negligible RD gains
but is currently done manually or at great computational
cost. Finally, for POTR-FT, we use two fine-tuning cycles
(RFT = 2) of 500 training iterations each. To ensure fairness,
comparisons to other fine-tuning methods match the total
number of training iterations.
4) Metrics: Our evaluation focuses predominately on RD
performance, with visual quality being objectively assessed
using PSNR, SSIM [77], and LPIPS [78]. We also report
inference time, though it is inherently hardware-dependent.
For a hardware-agnostic indicator of computational cost, we
additionally consider the number of splats.
B. Codec results
Table I and Figure 6 respectively present the quantitative and
qualitative performance of POTR(-FT). POTR achieves com-
pression ratios of 20–45× while retaining the models’ quality,
surpassing all other post-training codecs. This highlights that
no fine-tuning is required to achieve strong RD performance.
Using fine-tuning, POTR-FT further improves POTR’s already
superior RD performance, reducing storage requirements by
35-50% for the Deep Blending dataset, as shown in Figure 7.
Notably, the same figure shows that POTR(-FT) surpasses the
baseline’s quality at sufficiently high q, even without fine-
tuning.
POTR(-FT)’s strong RD performance primarily results from
our superior pruning method as our compressed models typi-
cally use more bytes per splat than other post-training meth-
ods. We suspect that the latter results from our comparatively
simple quantization, serialization, and entropy compression
3https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/
pretrained/models.zip

<!-- page 8 -->
8
Ground-truth
Size
# of splats
# of splats
Size
# of splats
Size
# of splats
Size
# of splats
Size
# of splats
Size
# of splats
Size
# of spl
Baseline
630 MB
2.54 M
MB
2.54 M
1.45 GB
5.83 M
1.45 GB
5.83 M
631 MB
2.55 M
631 MB
2.55 M
85.0 MB
343 k
85.0 MB
34
No fine-tuning
POTR (ours)
12.5 MB
822 k
MB
822 k
40.7 MB
2.36 M
40.7 MB
2.36 M
13.7 MB
635 k
13.7 MB
635 k
4.00 MB
192 k
4.00 MB
192
MesonGS
22.5 MB
1.52 M
MB
1.52 M
57.6 MB
4.20 M
57.6 MB
4.20 M
29.0 MB
2.04 M
29.0 MB
2.04 M
4.42 MB
274 k
4.42 MB
274
Fine-tuning
POTR-FT (ours)
10.2 MB
691 k
MB
691 k
31.7 MB
1.94 M
31.7 MB
1.94 M
9.64 MB
466 k
9.64 MB
466 k
3.22 MB
161 k
3.22 MB
16
MesonGS-FT
22.5 MB
1.52 M
57.6 MB
4.20 M
57.6 MB
4.20 M
29.0 MB
2.04 M
4.42 MB
274 k
4.42 MB
274
C3DGS
22.4 MB
2.05 M
49.1 MB
5.32 M
22.8 MB
2.28 M
4.99 MB
333 k
4.99 MB
33
Fig. 6:
Qualitative comparison of our proposed codec (POTR and POTR-FT) with the baseline (Kerbl et al. [11]) and the
state-of-the-art in post-training 3DGS compression (MesonGS, MesonGS-FT [46], and C3DGS [47]). One scene per discussed
dataset is considered, namely (from left to right): Truck (Tanks And Temples), Garden (Mip-NeRF 360), Playroom (Deep
Blending), and Lego (NeRF-Synthetic). Every model is annotated with its size and the number of splats.

<!-- page 9 -->
9
Method
Lossless
Format
FT
Tanks And Temples
Mip-NeRF 360
Deep Blending
NeRF-Synthetic
PSNR↑
SSIM↑LPIPS↓
Size
MB ↓# Splats
x1,000 ↓
PSNR↑
SSIM↑LPIPS↓
Size
MB ↓# Splats
x1,000 ↓
PSNR↑
SSIM↑LPIPS↓
Size
MB ↓# Splats
x1,000 ↓
PSNR↑
SSIM↑LPIPS↓
Size
MB ↓# Splats
x1,000 ↓
Baseline [11]
-
-
23.36 .838 .186 442 1,784 27.47 .821 .206 834 3,362 29.43 .898 .246 738 2,976 33.79 .970 .030 71.8
289
MesonGS [46]
npz
No 22.84 .820 .211 17.3 1,163 26.22 .785 .249 29.7 2,147 28.70 .890 .271 29.0 2,023 32.49 .962 .039 3.52
210
MesonGS-FT [46]
npz
Yes 23.16 .832 .200 17.3 1,163 27.03 .805 .231 29.7 2,147 29.54 .901 .255 29.0 2,023 33.25 .966 .035 3.51
210
C3DGS [47]
npz
Yes 23.13 .834 .195 18.3 1,483 27.16 .811 .226 30.4 2,973 29.35 .901 .256 26.7 2,613 33.31 .968 .033 3.99
270
LightGaussian [31]
npz
Yes 22.86 .817 .215 29.1
607
26.75 .805 .244 54.5 1,143 29.16 .894 .261 47.9 1,012 31.33 .956 .047 4.92
99
POTR (ours)
npz
No 23.27 .834 .191 12.8
690
27.08 .806 .226 29.3 1,500 29.31 .897 .253 18.6
785
33.34 .968 .032 3.94
154
zstd
11.3
26.0
16.5
3.50
POTR-FT (ours)
npz
Yes 23.34 .837 .189 10.6
594
27.20 .808 .223 24.1 1,285 29.44 .902 .250 13.2
585
33.37 .968 .032 3.12
126
zstd
9.36
21.3
11.8
2.82
TABLE I:
Quantitative comparison of our proposed codec with the baseline (uncompressed starting scene) and other
compression methods. The first , second , and third best compression results are highlighted for each metric across different
datasets. The ’FT’ column indicates whether a method utilizes fine-tuning. All fine-tuning methods use 1,000 training iterations.
The ’Lossless Format’ column indicates the lossless compression method used: NumPy’s (npz) or Meta’s Zstandard (zstd).
Parameter
Value
Parameter
Value
λ
10−q
SFSH
1 + 100q
α
sigmoid(3q)
SFopacity
1 + 200q
△MSEMAX
10−8.8−2q
SFrotation
1 + 400q
β
1.4 · 10−4 · q
SFscale
1 + 4000q
γ
5.0 · 10−3−5q
# pruning iterations
48
TABLE II: Hyperparameters as functions of q ∈[0, 1].
0
5
10
20
30
40
0.24
0.26
0.28
0.3
4x smaller
MesonGS
MesonGS-FT
C3DGS
LightGaussian
Size (MB)
LPIPS
POTR (ours)
POTR-FT (ours)
738
Baseline
Fig. 7: RD performance comparison of various post-training
methods on the Deep Blending dataset. Circle markers (◦) in-
dicate methods that utilize fine-tuning, while triangle markers
(△) denote methods without fine-tuning.
scheme. We believe future works could combine our ideas
with existing literature to further improve RD performance.
C. Pruning
Existing pruning methods are typically just as capable as
POTR at removing trivial and low-importance splats. It is
primarily after these splats are removed that our pruning
method starts to meaningfully outperform other methods. As
a result, our method keeps becoming comparatively better
than other methods as more and more distortion is tolerated.
To illustrate this, Figure 8 shows the relation between the
PSNR and the fraction of splats removed for various pruning
methods. Here, all methods remove the first 40% of splats
quasi-losslessly, but afterward, performances start to diverge
significantly. For a PSNR of 27, POTR requires 29% fewer
splats than MesonGS and LightGaussian, but for a PSNR of
40%
50%
60%
70%
80%
90%
100%
14
16
18
20
22
24
26
28
1.95x faster
177
fps
345
fps
1.61x faster
137
fps
221
fps
1.42x faster
120
fps
170
fps
No pruning (85 fps)
% of splats removed
PSNR
POTR (ours)
MesonGS(-FT)
LightGaussian
Fig. 8:
Quantitative comparison of post-training pruning
methods for the Garden model. Frame rate values represent
rendering at 1080p on an RTX 2080 Ti. LightGaussian and
MesonGS inference speeds are shown as one as they are quasi-
equal.
22, this has more than doubled to POTR requiring 64% fewer
splats.
Figure 8 also shows that our pruning method leads to
markedly faster inference times. While this is primarily due
to POTR simply removing more splats, there is a less obvious
secondary phenomenon at play as models created by POTR
can have far more splats, while still achieving the same
inference speed as other methods. For example, Figure 8 shows
that LightGaussian and MesonGS reach 177 fps after removing
75% of splats while POTR only has to remove a little over
65% of splats to achieve the same inference speed. Upon closer
inspection, we find that despite our model having more splats,
it has fewer Gaussian instances during rendering for common
camera poses. We suspect that this is due to our method
removing more splats that are more frequently in view, but
further research is required to verify this.
Beyond RD performance and inference speed, various other
topics benefit from having a better removal order. Examples

<!-- page 10 -->
10
0.7
0.8
0.9
1
DC only
No energy compaction
Sparsity AC coef.
α = 1.0
α = 0.9
α = 0.7
0
2
4
6
No energy compaction
Avg. L1-norm
non-zero AC coef.
−2
−1
0
1
2
3
4
5
23
24
25
No energy compaction
DC only (λ = +∞)
DC only (naive)
log10 λ
PSNR
Fig. 9: Effect of λ and α on the magnitude and sparsity of
the AC lighting coefficients and PSNR for the Truck model
(post-pruning and post-quantization). ’Naive’ refers to setting
all AC lighting coefficients to zero without recomputing the
DC lighting coefficients.
include level-of-detail rendering and progressive loading more
broadly. Future works could look at how our proposed pruning
method can be adapted to these purposes.
D. Spherical harmonics energy compaction
To better understand the effectiveness of spherical harmon-
ics energy compaction at lowering the entropy of the AC
lighting coefficients, we analyze how λ and α influence the
magnitude and sparsity of the AC lighting coefficients and
quality of the model. Figure 9 shows that the sparsity increases
far more quickly than the PSNR decreases upon applying
harsher regularization. This is despite the magnitude of non-
zero AC lighting coefficients remaining relatively constant,
indicating that by carefully choosing λ and α, the entropy of
the AC lighting coefficients can be dramatically reduced while
minimally impacting the PSNR. For example in Figure 9 for
λ = 101.1 and α = 0.9, the number of non-zero AC coef-
ficients decreases 10-fold. Despite eliminating 90% of non-
zero AC coefficients, the PSNR is minorly impacted, going
from 24.97 to 24.82. Further removing the remaining 10% of
AC coefficients would cause an 11-fold larger increase in the
MSE, highlighting our method’s effectiveness in identifying
Before energy compaction
After energy compaction
Pre-compaction
       L2-norm = 0.374
Zero fraction = 15/45
Post-compaction
       L2-norm = 0.190
Zero fraction = 22/45
.
Fig. 10: A blurred, top-down view of the Truck model shows
a red cross marking a splat from the truck’s red reflector.
Each training camera s is represented by a black arrow,
indicating its viewing direction, and a green dot whose size is
proportional to Ik(s). On the bottom, Mollweide projections
depict the splat’s view-dependent color before and after energy
compaction, where every green dot again corresponds with a
camera. Energy compaction results in a lower energy alias
that removes a hallucinated blue color (highlighted by a blue
rectangle) from an unsampled viewing area.
each lighting coefficient’s importance to the quality of the
model.
To practically illustrate the impact of energy compaction, we
examine a single splat in Figure 10. Here, training cameras
with high importance preserve their directions’ colors while
other directions’ colors change in favor of a low-energy
alias. As low-energy aliases typically correspond to ’simpler’
solutions, we indirectly follow Occam’s razor which often
leads to more sensible generalizations for unseen viewpoints.
Originally the splat in the red reflector became blue when
looked at from below, but with the simpler alias, the red
reflector stays red. More broadly, we suspect low-energy
aliases to be the driving force behind the PSNR increasing
slightly, from 24.92 to 24.97 under minimal regularization in
Figure 9. Notably, this increase in PSNR is achieved without
using any ground-truth images.
A key advantage of our energy compaction approach is its
adaptability to model- and scene-specific characteristics. For
example, the geometry of the scene and the camera trajectory
are clearly reflected in the sparsity of the spherical harmonics
bases. For the Truck model, splats are primarily observed from
the side. Low-energy solutions prefer spherical harmonics
bases that change significantly across samples as this leads
to smaller coefficients. As a result, the energy compacted
Truck model has three times fewer non-zero L2 than non-

<!-- page 11 -->
11
PSNR↑SSIM↑LPIPS↓Size
MB↓# Splats
x1,000 ↓
3DGS baseline
25.46
.775
.210
1,521
6,132
Ours
+ Serialization & Quantization
25.10
.760
.213
715.5
6,132
+ Entropy compression
25.10
.760
.213
189.3
6,132
+ Pruning
25.05
.755
.220
91.44
2,656
+ SH compaction
25.03
.754
.221
57.40
2,626
+ RGB →YCoCg
25.01
.753
.224
41.53
2,624
+ Fine-tuning
25.19
.758
.219
33.86
2,221
TABLE III:
Ablation study for the different components of
the POTR(-FT) encoder for the Bicycle model.
Splat property
Bytes per splat (%)
Compression
factor
Uncompressed
Compressed
Position
12 (5.1%)
3.58 (23 %)
3.35x
Scale
12 (5.1%)
2.40 (16 %)
6.45x
Opacity
4 (1.7%)
0.72 (4.7 %)
5.59x
Rotation
16 (6.8%)
3.29 (22 %)
4.87x
DC SH coefs.
12 (5.1%)
2.24 (15 %)
5.37x
AC SH coefs.
180 (76%)
3.03 (20 %)
59.4x
Total
236 (100%)
15.3 (100%)
15.5x
TABLE IV: Effect of the post-pruning steps (no fine-tuning)
on the size of the splat properties of the Bicycle model.
A property’s compressed size is approximated through its
ablation from the bitstream.
zero L4 values as Y2 remains relatively constant near the
’equator’, while Y4 changes substantially. Such observations
imply further predictability of the lighting coefficients and
are therefore useful for compression. POTR exploits these
observations through its serialization order, but future works
could exploit these observations more explicitly through, for
example, a custom entropy model.
E. Ablation Study
Table III presents an ablation study that evaluates the in-
cremental impact of the different components of our proposed
codec. It reveals that quantization, combined with our serial-
ization strategy and entropy compression, already compresses
the Bicycle model 8×. Incorporating pruning further reduces
the size by half, as more than half of the splats are removed.
Spherical harmonics energy compaction provides an additional
37% size reduction while only modifying lighting coefficients
and with virtually no quality loss. Similarly, switching from
an RGB to a YCoCg color representation (including more
harshly regularizing the chrominance channels) yields a further
28% reduction, again highlighting the importance of lighting
coeficients to 3DGS compression.
Finally, fine-tuning consistently improves all metrics, reduc-
ing both distortion and the number of splats. This improvement
arises because encoding and fine-tuning act in complemen-
tary ways: encoding is coarse, lossy, and approximate (e.g.,
pruning, quantization, and SH compaction), while fine-tuning
corrects the resulting errors through small, precise, global
adjustments. Iteratively alternating between encoding and fine-
tuning can hence yield further gains, as each process creates
additional rate-distortion improvement opportunities for the
other. For instance, SH energy compaction recomputes all
SH coefficients in a coarse, non-gradual manner based on
ficus
mic
materials
hotdog
drums
lego
ship
chair
truck
train
playroom
stump
treehill
flowers
drjohnson
counter
bonsai
garden
room
bicycle
kitchen
0
50
100
150
POTR Encoding Time (s)
Pruning
Entropy compression (zstd)
SH energy compaction
Miscellaneous
Fig. 11:
POTR’s encoding time for various models. ’Miscel-
laneous’ includes the remaining encoding operations, such as
quantization and serialization.
approximate statistics. Fine-tuning then refines these coeffi-
cients holistically, applying small, accurate, and globally con-
sistent corrections. In effect, fine-tuning validates the updates
introduced by energy compaction, allowing the subsequent
compaction step to be even more aggressive relative to the
original SH coefficients.
To quantify the compression of lighting coefficients, we
conduct a second ablation study, shown in Table IV, to
evaluate the contribution of individual splat properties to
the final compressed model size. We estimate a property’s
contribution to the file size by measuring the reduction in
the compressed model size when that property is removed
from the bitstream4. Table IV reveals that the AC lighting
coefficients are compressed by a factor of nearly 60×, whereas
other properties achieve more modest compression factors of
3-6×. Consequently, AC lighting coefficients are no longer the
primary contributors to the model size, with positional and
rotational information each occupying a larger share.
F. Codec speed
POTR and POTR-FT make trade-offs between their speed
and RD performance. For the results presented in Table I and
Figure 6, RD performance is prioritized, however, POTR can
still be fast. In this subsection we discuss the speed of our
codec using an NVIDIA GeForce RTX 5090 and AMD Ryzen
9 9950X.
Decoding is extremely fast, with all but the largest models
loading in one second or less. A linear model that fits the
total decoding time in seconds y as a function of the number
of splats post-pruning x with high accuracy is
y = 5.51 × 10−7x −0.02
(r > 0.999) .
This corresponds to a decoding throughput of approximately
1.8 million splats per second, which is sufficient for most
4Zstd’s compressed bitstream can, strictly speaking, not be decomposed into
distinct parts due to the holistic nature of lossless compression. However, this
analysis only aims to provide an approximate understanding of each property’s
contribution to the compressed file size.

<!-- page 12 -->
12
40%
50%
60%
70%
80%
90%
100%
14
16
18
20
22
24
26
28
1
4 81632
No pruning
% of splats removed
PSNR
1 iter ∼2.9s
4 iters ∼8.5s
8 iters ∼16s
16 iters ∼32s
32 iters ∼62s
Fig. 12:
The impact of limiting the number of pruning
iterations on the Pareto front of POTR’s pruning method for
the Garden model. Beyond 32 iterations (∼62 seconds), the
Pareto front still improves, but minorly.
practical use cases. The total decoding time is dominated by
three operations: zstd decompression, octree deserialization,
and attribute deserialization, which on average correspond to
10%, 62%, and 26% of total decode time respectively.
As shown in Figure 11, encoding is considerably slower,
with POTR requiring between 20 and 170, seconds depend-
ing on the model. Because POTR-FT repeatedly performs
encoding and its fine-tuning stage is comparatively fast, the
total encoding time of POTR-FT is closely tied to that of
POTR. Each of the RFT fine-tuning cycles takes roughly
80% of the original encoding time, although this can be
reduced substantially by integrating fine-tuning directly into
the training loop. Figure 11 also breaks down the contribution
of each encoding step to the overall encoding time, we discuss
the three most important steps in order of their importance:
1) Pruning: Pruning is typically the most time-consuming
step during encoding as every pruning iteration starts by recal-
culating all △MSEj values by executing a modified forward
pass per training view. Pruning time is therefore roughly linear
in
• the number of training views,
• the time per modified forward pass, and
• the number of pruning iterations.
We expect that ideas from sparse scene reconstruction [79],
[80] could be used to reduce the number of views rendered
by an order of magnitude. Additionally, we expect that the time
per modified forward pass can be further improved by carefully
optimizing our modified forward pass. We leave improving and
studying these ideas to future works and focus on the effect
of reducing the number of pruning iterations. Figure 12 shows
that the number of pruning iterations can be significantly
reduced if a slightly lower RD performance is tolerated. Even
using just one pruning iteration, our pruning performance
is already on par with LightGaussian and MesonGS (also
see Figure 8). Using only four pruning iterations, POTR
outperforms all other pruning strategies. We use 48 pruning
iterations (see Table II), which effectively guarantees that the
resulting RD performance is near-optimal.
2) Entropy compression : At the highest compression level,
entropy compression (zstd) can be a major contributor to
total runtime, especially for the larger models, since zstd
encode time scales roughly linearly with the final splat count.
Lowering the compression level can reduce this overhead to
the point of being negligible. For instance, using zstd at level
4 instead of level 22 yields approximately a 100× speedup, at
the cost of only about a 10% increase in file size.
3) Spherical harmonics energy compaction : Our proposed
spherical harmonics energy compaction method is embar-
rassingly parallel as each splat is processed independently.
Furthermore, by using ridge regression, a closed-form solution
is available that allows one of the 32 cores of our test machine
to energy compact upto 14,000 splats per second. Using a
sufficient number of threads thus makes the time spherical
harmonics energy compaction takes inconsequential.
VI. CONCLUSION
This work introduced POTR, a post-training codec for 3D
Gaussian Splatting that focuses on achieving strong rate-
distortion performance without fine-tuning. We first identified
that current post-training pruning methods rely heavily on
heuristics which often leads to suboptimal splat removal
decisions. To address this, we proposed an efficient method
to precisely compute the impact of each splat’s removal on
an objective quality metric. POTR leverages this knowledge
to remove splats in a manner that far outperforms existing
pruning methods, especially at higher distortion levels. Addi-
tionally, we proposed the first method to non-trivially reduce
the entropy of 3DGS lighting coefficients without training.
Our approach is fast, embarrassingly parallel, highly adaptable,
and shown to generalize lighting information more sensibly.
Combined with a simple quantization, serialization, and en-
tropy compression scheme, these innovations allow POTR
to significantly outperform existing methods in both rate-
distortion performance and inference speed, despite not using
fine-tuning. Moreover, we demonstrated that incorporating
a simple fine-tuning scheme further enhances POTR’s rate-
distortion performance and inference speed. We believe that
our codec’s smaller file sizes and faster inference speeds could
help make 3D Gaussian Splatting models more accessible,
especially for on-demand and virtual reality applications.
REFERENCES
[1] R.-S. Wang and Y. Wang, “Multiview video sequence analysis, compres-
sion, and virtual viewpoint synthesis,” IEEE Transactions on Circuits
and Systems for Video Technology, vol. 10, no. 3, pp. 397–410, 2000.
[2] J. Lu, S. Rogmans, G. Lafruit, and F. Catthoor, “Stream-centric stereo
matching and view synthesis: A high-speed approach on gpus,” IEEE
Transactions on Circuits and Systems for Video Technology, vol. 19,
no. 11, pp. 1598–1611, 2009.
[3] C. Song, S. Wang, J. Wei, and Y. Zhao, “Fewarnet: An efficient
few-shot view synthesis network based on trend regularization,” IEEE
Transactions on Circuits and Systems for Video Technology, vol. 34,
no. 10, pp. 9264–9280, 2024.
[4] K. Yamamoto, M. Kitahara, H. Kimata, T. Yendo, T. Fujii, M. Tanimoto,
S. Shimizu, K. Kamikura, and Y. Yashima, “Multiview video coding
using view interpolation and color correction,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 17, no. 11, pp. 1436–
1449, 2007.

<!-- page 13 -->
13
[5] A. I. Purica, E. G. Mora, B. Pesquet-Popescu, M. Cagnazzo, and
B. Ionescu, “Multiview plus depth video coding with temporal prediction
view synthesis,” IEEE Transactions on Circuits and Systems for Video
Technology, vol. 26, no. 2, pp. 360–374, 2016.
[6] X. Tian, J. Shao, D. Ouyang, and H. T. Shen, “Uav-satellite view syn-
thesis for cross-view geo-localization,” IEEE Transactions on Circuits
and Systems for Video Technology, vol. 32, no. 7, pp. 4804–4815, 2022.
[7] R. Verhack, T. Sikora, G. Van Wallendael, and P. Lambert, “Steered
mixture-of-experts for light field images and video: Representation and
coding,” IEEE Transactions on Multimedia, vol. 22, no. 3, pp. 579–593,
2020.
[8] J. Artois, M. Courteaux, G. Wallendael, and P. Lambert, “Opendibr:
Open real-time depth-image-based renderer of light field videos for vr,”
Multimedia Tools and Applications, vol. 83, pp. 25 797–25 815, 08 2023.
[9] M. Courteaux, H. Mareen, B. Ramlot, P. Lambert, and G. Van Wal-
lendael, “Dimensionality reduction for the real-time light-field view
synthesis of kernel-based models,” Electronics, vol. 13, no. 20, 2024.
[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” in ECCV, 2020.
[11] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, July 2023.
[12] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” ICCV, 2021.
[13] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” CVPR,
2022.
[14] ——, “Zip-nerf: Anti-aliased grid-based neural radiance fields,” ICCV,
2023.
[15] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and
G. Drettakis, “A hierarchical 3d gaussian representation for real-time
rendering of very large datasets,” ACM Transactions on Graphics,
vol. 43, no. 4, July 2024.
[16] Y. Liu, H. Guan, C. Luo, L. Fan, J. Peng, and Z. Zhang, “Citygaussian:
Real-time high-quality large-scale scene rendering with gaussians,”
2024.
[17] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian,
and W. Xinggang, “4d gaussian splatting for real-time dynamic scene
rendering,” arXiv preprint arXiv:2310.08528, 2023.
[18] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d
gaussians: Tracking by persistent dynamic view synthesis,” in 3DV,
2024.
[19] Y. Zhang, W. Jia, W. Niu, and M. Yin, “Gaussianspa: An” optimizing-
sparsifying” simplification framework for compact and high-quality 3d
gaussian splatting,” in Proceedings of the Computer Vision and Pattern
Recognition Conference, 2025, pp. 26 673–26 682.
[20] M. Wu and T. Tuytelaars, “Implicit gaussian splatting with efficient
multi-level tri-plane representation,” 2024.
[21] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, “Compression of
3d gaussian splatting with optimized feature planes and standard video
codecs,” arXiv preprint arXiv:2501.03399, 2025.
[22] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d
scene representation via self-organizing gaussian grids,” arXiv preprint
arXiv:2312.13299, 2023.
[23] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, “Compgs:
Efficient 3d scene representation via compressed gaussian splatting,” in
Proceedings of the 32nd ACM International Conference on Multimedia,
2024, pp. 2936–2944.
[24] X. Liu, X. Wu, S. Wang, Z. Li, and S. Kwong, “Compgs++: Compressed
gaussian splatting for static and dynamic scene representation,” arXiv
preprint arXiv:2504.13022, 2025.
[25] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid
assisted context for 3d gaussian splatting compression,” in European
Conference on Computer Vision, 2024.
[26] ——, “Hac++: Towards 100x compression of 3d gaussian splatting,”
arXiv preprint arXiv:2501.12255, 2025.
[27] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash,
“Compact3d: Smaller and faster gaussian splatting with vector quanti-
zation,” arXiv preprint arXiv:2311.18159, 2023.
[28] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, “End-
to-end rate-distortion optimized 3d gaussian representation,” 2024.
[29] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated
3d gaussians with lightweight encodings,” 2024.
[30] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[31] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2024,
pp. 21 719–21 728.
[32] X. Sun, J. C. Lee, D. Rho, J. H. Ko, U. Ali, and E. Park, “F-3dgs:
Factorized coordinates and representations for 3d gaussian splatting,”
arXiv preprint arXiv:2405.17083, 2024.
[33] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis,
“Reducing the memory footprint of 3d gaussian splatting,” Proceedings
of the ACM on Computer Graphics and Interactive Techniques, vol. 7,
no. 1, May 2024.
[34] Y. Wang, Z. Li, L. Guo, W. Yang, A. Kot, and B. Wen, “Contextgs: Com-
pact 3d gaussian splatting with anchor level context model,” Advances
in neural information processing systems, vol. 37, pp. 51 532–51 551,
2024.
[35] S. Shin, J. Park, and S. Cho, “Locality-aware gaussian compression for
fast and high-quality rendering,” arXiv preprint arXiv:2501.05757, 2025.
[36] J. Cao, V. Goel, C. Wang, A. Kag, J. Hu, S. Korolev, C. Jiang,
S. Tulyakov, and J. Ren, “Lightweight predictive 3d gaussian splats,”
arXiv preprint arXiv:2406.19434, 2024.
[37] G. Fang and B. Wang, “Mini-splatting: Representing scenes with a
constrained number of gaussians,” in European Conference on Computer
Vision.
Springer, 2024, pp. 165–181.
[38] Z. Zhang, T. Song, Y. Lee, L. Yang, C. Peng, R. Chellappa, and D. Fan,
“Lp-3dgs: Learning to prune 3d gaussian splatting,” Advances in Neural
Information Processing Systems, vol. 37, pp. 122 434–122 457, 2024.
[39] A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, and T. Goldstein,
“Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse
primitives,” in Proceedings of the Computer Vision and Pattern Recog-
nition Conference, 2025, pp. 21 537–21 546.
[40] M. Niemeyer, F. Manhardt, M.-J. Rakotosaona, M. Oechsle, D. Duck-
worth, R. Gosula, K. Tateno, J. Bates, D. Kaeser, and F. Tombari,
“Radsplat: Radiance field-informed gaussian splatting for robust real-
time rendering with 900+ fps,” in 2025 International Conference on 3D
Vision (3DV).
IEEE, 2025, pp. 134–144.
[41] G.
Fang
and
B.
Wang,
“Mini-splatting2:
Building
360
scenes
within minutes via aggressive gaussian densification,” arXiv preprint
arXiv:2411.12788, 2024.
[42] Y.-T. Zhan, C.-Y. Ho, H. Yang, Y.-H. Chen, J. C. Chiang, Y.-L.
Liu, and W.-H. Peng, “Cat-3dgs: A context-adaptive triplane ap-
proach to rate-distortion-optimized 3dgs compression,” arXiv preprint
arXiv:2503.00357, 2025.
[43] Y. Chen, M. Li, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Pcgs:
Progressive compression of 3d gaussian splatting,” arXiv preprint
arXiv:2503.08511, 2025.
[44] J. Chen, Y. Chen, Y. Zou, Y. Huang, P. Wang, Y. Liu, Y. Sun, and
W. Wang, “Megs2: Memory-efficient gaussian splatting via spherical
gaussians and unified pruning,” arXiv preprint arXiv:2509.07021, 2025.
[45] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
2023.
[46] S. Xie, W. Zhang, C. Tang, Y. Bai, R. Lu, S. Ge, and Z. Wang, “Mesongs:
Post-training compression of 3d gaussians via efficient attribute trans-
formation,” in European Conference on Computer Vision.
Springer,
2024.
[47] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR), June 2024, pp. 10 349–10 358.
[48] Y. Lee, Z. Zhang, and D. Fan, “Safeguardgs: 3d gaussian primitive
pruning while avoiding catastrophic scene destruction,” arXiv preprint
arXiv:2405.17793, 2024.
[49] A. Hanson, A. Tu, V. Singla, M. Jayawardhana, M. Zwicker, and
T. Goldstein, “Pup 3d-gs: Principled uncertainty pruning for 3d gaussian
splatting,” in Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, 2025, pp. 5949–5958.
[50] Y. Huang, J. Pang, F. Zhu, and D. Tian, “Entropygs: An efficient entropy
coding on 3d gaussian splatting,” arXiv preprint arXiv:2508.10227,
2025.
[51] W. Liu, T. Guan, B. Zhu, L. Xu, Z. Song, D. Li, Y. Wang, and
W. Yang, “Efficientgs: Streamlining gaussian splatting for large-scale
high-resolution scene representation,” IEEE MultiMedia, 2025.

<!-- page 14 -->
14
[52] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph.,
vol. 41, no. 4, pp. 102:1–102:15, Jul. 2022.
[53] C. Reiser, S. Peng, Y. Liao, and A. Geiger, “Kilonerf: Speeding up neural
radiance fields with thousands of tiny mlps,” in International Conference
on Computer Vision (ICCV), 2021.
[54] C. L. Deng and E. Tartaglione, “Compressing explicit voxel grid
representations: Fast nerfs become also small,” in Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision
(WACV), January 2023, pp. 1236–1245.
[55] L. Li, Z. Shen, Z. Wang, L. Shen, and L. Bo, “Compressing volumetric
radiance fields to 1 mb,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), June 2023, pp.
4222–4231.
[56] T.
Zhao,
J.
Chen,
C.
Leng,
and
J.
Cheng,
“Tinynerf:
to-
wards 100× compression of voxel radiance fields,” in AAAI, ser.
AAAI’23/IAAI’23/EAAI’23.
AAAI Press, 2023.
[57] D. Rho, B. Lee, S. Nam, J. C. Lee, J. H. Ko, and E. Park, “Masked
wavelet representation for compact neural radiance fields,” in Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 20 680–20 690.
[58] S. Li, H. Li, Y. Liao, and L. Yu, “Nerfcodec: Neural feature compression
meets neural radiance fields for memory-efficient scene representation,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 21 274–21 283.
[59] T. Pham and S. Mandt, “Neural nerf compression,” in Proceedings of the
41st International Conference on Machine Learning, 2024, pp. 40 592–
40 610.
[60] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, “Ecrf: Entropy-
constrained neural radiance fields compression with frequency domain
optimization,” in 2024 IEEE 26th International Workshop on Multimedia
Signal Processing (MMSP).
IEEE, 2024, pp. 1–6.
[61] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu,
“Neural residual radiance fields for streamably free-viewpoint videos,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 76–87.
[62] T. Takikawa, A. Evans, J. Tremblay, T. M¨uller, M. McGuire, A. Jacob-
son, and S. Fidler, “Variable bitrate neural fields,” in ACM SIGGRAPH
2022 Conference Proceedings, ser. SIGGRAPH ’22.
New York, NY,
USA: Association for Computing Machinery, 2022.
[63] S. Shin and J. Park, “Binary radiance fields,” in Proceedings of the 37th
International Conference on Neural Information Processing Systems, ser.
NIPS ’23.
Red Hook, NY, USA: Curran Associates Inc., 2024.
[64] Y. Chen, Q. Wu, M. Harandi, and J. Cai, “How far can we compress
instant-ngp-based nerf?” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), June 2024, pp.
20 321–20 330.
[65] Z. Wang, S. Wan, and L. Wei, “Local geometry-based intra prediction for
octree-structured geometry coding of point clouds,” IEEE Transactions
on Circuits and Systems for Video Technology, vol. 33, no. 2, pp. 886–
896, 2023.
[66] Z. Que, G. Lu, and D. Xu, “Voxelcontext-net: An octree based frame-
work for point cloud compression,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June
2021, pp. 6042–6051.
[67] L. Huang, S. Wang, K. Wong, J. Liu, and R. Urtasun, “Octsqueeze:
Octree-structured entropy model for lidar compression,” in 2020
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2020, pp. 1310–1320.
[68] C. Cao, M. Preda, and T. Zaharia, “3d point cloud compression: A
survey,” in Proceedings of the 24th International Conference on 3D
Web Technology, ser. Web3D ’19.
New York, NY, USA: Association
for Computing Machinery, 2019, p. 1–9.
[69] R. L. de Queiroz and P. A. Chou, “Compression of 3d point clouds
using a region-adaptive hierarchical transform,” IEEE Transactions on
Image Processing, vol. 25, no. 8, pp. 3947–3956, 2016.
[70] Y. Huang, J. Peng, C.-C. J. Kuo, and M. Gopi, “Octree-Based Progres-
sive Geometry Coding of Point Clouds,” in Symposium on Point-Based
Graphics, M. Botsch, B. Chen, M. Pauly, and M. Zwicker, Eds.
The
Eurographics Association, 2006.
[71] K. Zhang, W. Zhu, and Y. Xu, “Hierarchical segmentation based point
cloud attribute compression,” in 2018 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 3131–
3135.
[72] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A.
Chou, R. A. Cohen, M. Krivoku´ca, S. Lasserre, Z. Li, J. Llach,
K. Mammou, R. Mekuria, O. Nakagami, E. Siahaan, A. Tabatabai, A. M.
Tourapis, and V. Zakharchenko, “Emerging mpeg standards for point
cloud compression,” IEEE Journal on Emerging and Selected Topics in
Circuits and Systems, vol. 9, no. 1, pp. 133–148, 2019.
[73] M. T. Bagdasarian, P. Knoll, F. Barthel, A. Hilsmann, P. Eisert, and
W. Morgenstern, “3dgs.zip: A survey on 3d gaussian splatting compres-
sion methods,” 2024.
[74] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Conference on Computer Vision and Pattern Recognition (CVPR),
2016.
[75] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Bros-
tow, “Deep blending for free-viewpoint image-based rendering,” ACM
Transactions on Graphics (SIGGRAPH Asia Conference Proceedings),
vol. 37, no. 6, November 2018.
[76] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics, vol. 36, no. 4, 2017.
[77] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assess-
ment: from error visibility to structural similarity,” IEEE Transactions
on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[78] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
2018 IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2018, pp. 586–595.
[79] Z. Liu, J. Su, G. Cai, Y. Chen, B. Zeng, and Z. Wang, “Georgs: Geomet-
ric regularization for real-time novel view synthesis from sparse inputs,”
IEEE Transactions on Circuits and Systems for Video Technology, pp.
1–1, 2024.
[80] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
“Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with
global-local depth normalization,” arXiv preprint arXiv:2403.06912,
2024.
Bert Ramlot received the M.Sc. degree in com-
puter science engineering from Ghent University,
Belgium, in 2024, where he is currently pursuing the
Ph.D. degree with IDLab, imec. His current research
interests are 3D Gaussian Splatting and compression
thereof.
Martijn Courteaux received the M.Sc. degree in
computer science engineering from Ghent Univer-
sity, Belgium, in 2018, where he is currently pur-
suing the Ph.D. degree with IDLab, imec, through
the financial support of the Research Founda-
tion–Flanders (FWO). His current research interests
focus on the modeling and compression of light
fields and light-field videos, and are set in the context
of statistics, signal processing, and compression.
Peter Lambert (Senior Member, IEEE) is Full
Professor at the IDLab of Ghent University – imec
(Belgium). He received his Master’s degree in sci-
ence (mathematics) and in applied informatics from
Ghent University in 2001 and 2002, respectively,
and he obtained the Ph.D. degree in computer sci-
ence in 2007 at the same university. His research
interests include multimedia signal processing, data
compression, computer graphics, XR, and visual
communications.

<!-- page 15 -->
15
Glenn Van Wallendael (Member, IEEE) obtained
the M.Sc. degree in Computer Science Engineer-
ing from Ghent University, Belgium in 2008. Af-
terwards, he obtained the Ph.D. at IDLab, Ghent
University, with the financial support of the Research
Foundation - Flanders (FWO). Since 2019, he works
as a Professor for both Ghent University and imec on
topics such as the efficient representation and com-
pression of visual information, including immersive
media, 360 degree video, light fields, and virtual
reality.
