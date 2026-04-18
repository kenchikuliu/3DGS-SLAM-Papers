<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
Improving 3D Gaussian Splatting Compression by Scene-Adaptive
Lattice Vector Quantization
Hao Xu, Xiaolin Wu, Life Fellow, IEEE, and Xi Zhang, Member, IEEE
Abstract—3D Gaussian Splatting (3DGS) is rapidly gaining
popularity for its photorealistic rendering quality and real-time
performance, but it generates massive amounts of data. Hence
compressing 3DGS data is necessary for the cost effectiveness
of 3DGS models. Recently, several anchor-based neural com-
pression methods have been proposed, achieving good 3DGS
compression performance. However, they all rely on uniform
scalar quantization (USQ) due to its simplicity. A tantalizing
question is whether more sophisticated quantizers can improve
the current 3DGS compression methods with very little extra
overhead and minimal change to the system. The answer is
yes by replacing USQ with lattice vector quantization (LVQ).
To better capture scene-specific characteristics, we optimize the
lattice basis for each scene, improving LVQ’s adaptability and R-
D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance
between the R-D efficiency of vector quantization and the low
complexity of USQ. SALVQ can be seamlessly integrated into
existing 3DGS compression architectures, enhancing their R-
D performance with minimal modifications and computational
overhead. Moreover, by scaling the lattice basis vectors, SALVQ
can dynamically adjust lattice density, enabling a single model to
accommodate multiple bit rate targets. This flexibility eliminates
the need to train separate models for different compression levels,
significantly reducing training time and memory consumption.
Index Terms—3DGS compression, lattice vector quantization,
variable rate data compression.
I. INTRODUCTION
N
OVEL view synthesis has witnessed remarkable progress
in recent years, with Neural Radiance Fields (NeRF) [2]
established as a significant milestone in the field. While NeRF
substantially improves rendering quality compared to its pre-
decessors [3]–[5], it suffers from slow rendering speeds due to
the high computational cost of querying density and radiance
values along camera rays. More recently, 3D Gaussian Splat-
ting (3DGS) [6] has rapidly gained popularity as an alternative
approach, offering both high-quality rendering results and real-
time performance. However, 3DGS requires a large number of
Gaussian primitives to accurately represent a 3D scene, and
storing these Gaussian attributes incurs substantial memory
consumption. This motivates the research on compression of
3DGS models.
This work is based on Chapter 4 of the Ph.D. thesis of Hao Xu [1], with
additional experiments and revisions for journal publication.
H.
Xu
is
with
the
Department
of
Electrical
&
Computer
Engi-
neering, McMaster University, Hamilton, ON L8S 4L8, Canada (email:
xu338@mcmaster.ca).
X. Wu is with the School of Computing and Artificial Intelligence,
Southwest Jiaotong University, Chengdu, China (Corresponding author, email:
xwu510@gmail.com).
X. Zhang is with the ANGEL Lab, Nanyang Technological University,
Singapore. (email: xi.zhang@ntu.edu.sg).
As redundancies exist in 3DGS models, some Gaussian
primitives may be noncritical and not all attributes of these
primitives require uniformly high precision to maintain ren-
dering quality. Pruning and quantization can be leveraged
to improve memory efficiency without materially sacrificing
rendering performance [7]–[12].
However, these methods are limited in removing spatial re-
dundancy because they largely ignore the correlations between
neighboring Gaussian primitives. To rectify this shortcoming,
Scaffold-GS introduces a hierarchical representation by using
a sparse set of anchors to generate a dense set of so-called
neural Gaussians [13]. Each anchor is associated with a group
of neural Gaussians whose positions are defined by learn-
able offsets. The attributes of these Gaussians (i.e. opacity,
color, rotation, scale) are dynamically predicted based on the
anchor features and the viewing direction. Scaffold-GS lays
the foundation for subsequent compression methods [14]–[18],
which employ quantization and context-adaptive arithmetic
coding to reduce the memory footprint of anchor primitives.
By estimating the conditional probability of each symbol given
its context, these models reduce statistical redundancy and
improve coding efficiency, representing the current state-of-
the-art (SOTA) performance in 3DGS compression.
A common drawback of anchor-based 3DGS compression
methods is that they adopt uniform scalar quantization (USQ)
to quantize anchor attributes. This naive scheme, although
simplifies implementation, significantly compromises the rate-
distortion (R-D) performance. Here a natural question arises:
can we design more sophisticated quantizers that can improve
the current SOTA methods with negligible overhead and minor
changes to the system. We answer the question affirmatively
by replacing USQ with lattice vector quantization (LVQ).
To better fit scene-specific feature distributions, we warp the
lattice shape with respect to each scene, adapting LVQ to the
scene structures and hence improving R-D efficiency. This
scene-adaptive LVQ (SALVQ) strikes a balance between the
coding efficiency of vector quantization (VQ) and the low
complexity of USQ. SALVQ can be seamlessly embedded
into existing 3DGS compression architectures with minimal
modification, enhancing their R-D performance with almost
no extra overhead. Moreover, by scaling the basis vectors
of SALVQ, the proposed neural 3DGS neural compression
model can dynamically adjust lattice density, enabling a single
model to accommodate multiple target bit rates. This flexibility
eliminates the need to train separate models for different
compression levels, significantly reducing training time and
memory consumption.
0000–0000/00$00.00 © 2021 IEEE
arXiv:2509.13482v1  [cs.CV]  16 Sep 2025

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
Position 𝐱
Anchor attributes:
1. Feature 𝒇
2. Scaling 𝒍
3. K learnable offsets 𝑶𝒊𝑖=1
𝐾
Each anchor generates
 𝐾 Gaussian primitives. 
Use three three tiny MLPs  to predict 
color, opacity and  covairance (scale 
and quaternions) for the generated 
Gaussian primitives of each anchor.
USQ
Entropy 
coding
Decompressed position ෝ𝒙
Decompressed attributes:
1. Feature ෠𝒇
2. Scaling መ𝒍
3. K learnable offsets ෡𝑶𝒊𝑖=1
𝐾
Multi-scale Hash grids
…….
Query, interpolate
and concatenate
MLP
(𝝁, 𝝈)
𝑞𝑠
Camera center 𝒙𝒄
(෠𝒇, ||ෝ𝒙−𝒙𝒄||2,
ෝ𝒙−𝒙𝒄
||ෝ𝒙−𝒙𝒄||2
)
Rendering
The compression pipeline.
The rendering pipeline of Scaffold-GS.
Ours SALVQ
Our contribution: replace USQ 
with SALVQ. 
Position of generated 
Gaussian primitives:
𝝁𝒊
𝒈
𝑖=1
𝐾
= ෝ𝒙+ ෡𝑶𝒊𝑖=1
𝐾
∙መ𝒍
Fig. 1: Illustrating the pipeline of anchor-based neural compression methods, using Hash-Assisted Context (HAC) [14] as an
example. Subsequent methods [15]–[18] follow this pipeline with different context models. Their superior R–D performance
largely comes from learning a scene-specific Scaffold-GS representation and context model, whereas the quantizer remains a
shared USQ. We go a step further by making the quantizer learnable and scene-adaptive so that its Voronoi partition better
matches scene-specific statistics.
In summary, our main contribution is to use the novel
scene-adaptive LVQ in 3DGS compression to improve the
storage and bandwidth economies of the 3DGS applications.
The proposed SALVQ approach can facilitate the growing
deployment of the 3DGS system, owing to its following
practical advantages:
• Adaptability to specific scene statistics by warping LVQ
cells to best fit the distribution of anchor features to be
coded.
• The ability to support variable-rate 3DGS compression,
offering flexibility in bitrate control while delivering
high-quality reconstruction results.
• Compatibility with all neural compression architectures,
meaning that the SALVQ method can be adopted with
only minor modifications to existing systems.
II. RELATED WORK
A. 3DGS and its compression
The emerging 3DGS [6] represents scenes using a set of
3D Gaussian primitives and employs an efficient rasterization
pipeline for rendering, achieving both high visual quality
and real-time performance. The memory footprint of a single
Gaussian primitive is determined by storing its attributes,
including position (3 floats), scale (3 floats), color (3 floats),
rotation (4 floats for a quaternion), opacity (1 float), and Spher-
ical Harmonics coefficients (45 floats), totaling 59 floats per
primitive. After training, the number of Gaussian primitives
can exceed millions, resulting in a memory footprint of several
hundred MB or even several GB per scene. Such large memory
consumption significantly restricts the deployment of 3DGS on
memory-constrained devices and leads to substantial storage
and transmission overhead. Thus, improving the memory ef-
ficiency of 3DGS has become a key research focus. Common
methods include pruning redundant 3D Gaussians [7]–[11],
[14], [19]–[26], reducing the degree of SH coefficients [7], [9],
[10], vector quantization [8]–[12] and attribute transform [19],
[20]. Furthermore, there is a growing consensus that ideas
from point cloud compression [27]–[34] can inform and in-
spire the development of efficient 3DGS compression.
B. Anchor-based GS and its compression
The methods discussed in Sec.II-A achieve only limited
compression ratios because they overlook the inherent spa-
tial organization among Gaussians, thus being classified as
unstructured compression techniques. In contrast, structured
compression methods explicitly exploit the spatial relation-
ships and hierarchical organization of Gaussian representations
for more efficient storage. The most prominent structured
compression approach is Scaffold-GS [13], which leverages
anchors as structured reference points. Each anchor is char-
acterized by its position x, latent feature f, scaling factor
l, and a set of K learnable offsets {Oi}K
i=1. Each anchors
generates K Gaussian primitives whose positions depend on
the anchor’s scaling factor and offsets. The attributes of each
Gaussian primitive (e.g., color, covariance, opacity) are then
predicted from the anchor feature f and camera position xc
using a small MLP. The right panel of Fig. 1 illustrates this
anchor-based generation process for K = 5.
Building upon the Scaffold-GS backbone [13], context
modeling techniques [14]–[18] have been incorporated into the
entropy coding stage to reduce statistical redundancy among
anchors, thereby achieving a more compact bitstream. Fig. 1
illustrates the pipeline used by these methods, using hash-
assisted context (HAC) [14] as an example. As a prominent
context modeling approach, HAC employs multi-scale hash
grids as a hyperprior to effectively guide the entropy coder.
Given the position of an anchor, HAC queries these hash
grids to retrieve contextual features, which are subsequently
processed by an MLP to predict the quantization scaling factor
qs, as well as the mean µ and standard deviation σ of the
Gaussian distribution for the anchor attributes. Building upon
HAC, several more complex and powerful entropy models
have been proposed. For example, HAC++ [17] and CAT-
3DGS [15] employ a channel-wise autoregressive model [35],
whereas ContextGS [18] and HEMGS [16] adopt spatial-wise
autoregressive models [36], [37] to further enhance compres-
sion performance.
Although these methods achieve the current SOTA perfor-
mance by learning a scene-adaptive Scaffold-GS represen-

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
x
y
x
y
x
y
x
y
1
2
r = 1
𝜋
2 3
9
2
2
ඵ(x2 + y2)dxdy ≈0.1667 ඵ(x2 + y2)dxdy ≈0.1592 ඵ(x2 + y2)dxdy ≈0.1604 ඵ(x2 + y2)dxdy ≈0.1667
(a) square cell
(b) circle cell
(c) hexagonal cell
(e) An example demonstrating LVQ’s ability 
to capture correlations. 
(d) diamond cell
x
y
2
1
( 2, 0)
( 2
2 ,
2
2 )
x
y
2
1
(1,0)
(1,1)
Fig. 2: Illustration of the advantages of LVQ over USQ. For equal cell area, lattices whose Voronoi cells more closely
approximate a circle yield a smaller mean-squared distance to the cell center than the canonical square (a–d). With equal-
area diamond and square cells and correlated source components, diamond LVQ shortens the nearest-neighbor inter-codeword
spacing along the principal direction, reducing the expected distortion; USQ leaves a larger spacing along the same principal
direction (e).
tation and a matching scene-adaptive context model, they
typically rely on USQ for simplicity. USQ’s limited ability to
accommodate diverse scene statistics leaves substantial room
to improve R–D performance. This motivates us to propose
a scene-adaptive quantizer that can be plugged into existing
pipelines to further improve R–D performance with minimal
modification to the backbone.
C. LVQ in neural image compression
Given the advantages of LVQ that LVQ achieves a cost-
effective balance between VQ and USQ, some researchers
have explored the use of LVQ in neural image compres-
sion [38]–[42]. These approaches typically improve R-D per-
formance with minimal modifications to existing network
architectures. Generally, they employ a common LVQ across
various images, with the lattice basis either being prede-
fined [38], [39] or optimized during training [41]. To en-
hance flexibility, Xu et al.proposed a joint strategy for rate
and domain adaptation by learning linear transforms of the
lattice basis matrix [42]. Specifically, they achieve rate control
by scaling the lattice basis vectors. For domain adaptation,
they apply an invertible linear transforms to modulate the
predefined lattice basis matrix, allowing it to better match the
distinct characteristics of different image categories.
III. LVQ PRELIMINARY
In preparation for the main technical development we in-
troduce the basics of LVQ, including its concept and inherent
coding advantages over scalar quantization.
Regular arrangements of points in vector space are called
lattices [43]. In Rn, a lattice is formed by taking all integer
linear combinations of a set of linearly independent basis
vectors {b1, b2, . . . , bn}, denoted by
Λ = {z|z = Bu =
n
X
i=1
uibi, ui ∈Z}
(1)
where B is the lattice basis matrix. A vector quantizer whose
codewords are restricted to be lattice points is called lattice
vector quantizer.
Thanks to the regular lattice structure, the nearest neighbor
encoding of LVQ in Rn becomes very simple and can be
performed in O(n) time independent of the codebook size L,
in contrast to the O(nL) time of conventional VQ [44].
Despite its regularity and low cost, LVQ still enjoys the
coding benefit of VQ. Compared to USQ, LVQ has two
inherent advantages: space-filling efficiency and the ability
to capture correlation [45]. Fig. 2 (a-d) illustrates the space-
filling advantage. Given the area/volume of a Voronoi cell,
the optimal cell shape in terms of minimal mean squared
quantization error is circle or hypersphere in high dimen-
sions. However, tiling the space with spheres cannot avoid
either overlaps or holes. LVQ addresses this issue by using
a carefully designed lattice to partition a vector space into
congruent convex Voronoi cells whose shape is as close to
sphere as possible, achieving most efficient space covering. For
some dimensions, optimal lattices are known to have Voronoi
cells that best approximate spheres [46]. Examples include the
hexagonal lattice (A2) in R2, as well as the E8 lattice in R8
and Leech lattice in R24.
Moreover, LVQ can capture the feature correlations in a way
that USQ cannot. Fig. 2 (e) depicts the case of correlation,
where teal and purple points represent input vectors and
their quantized codewords, respectively. In the diamond LVQ,
the distance between two nearest quantizer codewords in the
principal (diagonal) direction is 1; in contrast, the adjacent
codeword distance in horizontal (non-principal) direction is
√
2. By reducing quantization error along the principal direc-
tion, LVQ lowers the expected distortion and thus improves
compression efficiency.
IV. METHOD
In this section, we first give an overview of LVQ design
process. We then detail our method for optimizing LVQ on
a per-scene basis and explain how it is integrated into an
anchor-based neural 3DGS compression architecture. Finally,
we develop a rate-control scheme that enables our LVQ-based
3DGS compression system to operate in variable-rate mode.
A. Overview
The anchor-based 3DGS compression approach presents the
current state of the art in the aspect of R–D performance.
This is achieved by aggressive scene-specific overfitting: a

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Scaffold-GS scene representation and a conditional probability
model for the entropy coding of anchor attributes are jointly
trained per scene. As a result, these two components are
highly tailored to each individual scene. Yet, as shown in
Fig. 1, the quantizer remains a simple, scene-agnostic uniform
scalar quantizer (USQ); the only adjustable parameter is the
step size. Because a USQ tiles the space with axis-aligned
hypercubes, its space-filling efficiency is low, compromising
compression performance. This naturally raises the question:
can we also overfit a learnable LVQ that is jointly optimized
with the Scaffold-GS and entropy model to further improve
R-D performance? Our answer is emphatically yes.
To investigate this affirmative answer in practice, we begin
by discussing the most direct and widely familiar approach:
overfitting a separate codebook for each scene and applying
VQ to the anchor attributes. In practice, however, conven-
tional VQ is problematic. The encoding stage of VQ is
non-differentiable. Although the soft-to-hard VQ [47] and
straight-through estimator (STE) [48]-based approaches [49]
can circumvent the non-differentiability issue, the former often
struggles with low training stability, and the latter is prone
to codebook collapse issue [50]. Moreover, VQ inherently
requires a nearest neighbor search during encoding, which in-
creases computational cost. In addition, it needs to store a large
per-scene codebook, which increases the memory footprint and
offsets the rate savings achieved by representing vectors with
codebook indices. These drawbacks become more pronounced
as the codebook size increases, making conventional VQ less
practical for efficient 3DGS compression.
In contrast, LVQ addresses several limitations of conven-
tional VQ. LVQ generates the codebook implicitly from a
learnable lattice basis matrix; only this compact basis needs
to be stored, not a large explicit codebook. Because all
code vectors are derived from a shared lattice structure, the
number of learnable parameters is much smaller than in
independently optimized codebooks, and this shared structure
imposes geometric constraints that improve training stability.
The lattice structure also admits fast quantization algorithms,
reducing encoding cost. Together, these properties make LVQ
a more practical and powerful replacement for USQ in 3DGS
compression. While fixed lattice quantizers such as Dn and E8
are easy to deploy, their scene-agnostic nature prevents them
from fully exploiting LVQ’s potential. By contrast, jointly
optimizing the lattice basis, Scaffold-GS, and the entropy
model within a scene-adaptive LVQ, though more difficult,
delivers greater returns. The following subsections detail the
design of the proposed scene-adaptive LVQ.
B. Learning scene-adaptive LVQ
When learning SALVQ for 3DGS compression, one might
consider leveraging two strategies previously explored in
neural image compression. The first strategy is to directly
optimize the lattice basis matrix while adding a regularization
term to promote the orthogonality of the basis vectors [41].
The second strategy factorizes the lattice basis matrix B
into the product of a learnable linear transform A and a
predefined lattice basis matrix G, i.e., B = AG [42]. In
this formulation, the linear transform A is constrained to be
invertible and is parameterized as A = VΣVT, where V is
a learnable rotation matrix derived from matrix exponential
mapping and Σ is a learnable diagonal matrix. However, both
strategies have notable drawbacks when directly applied to
3DGS compression. The first strategy may lead to training
instability and can occasionally result in a non-invertible lattice
basis matrix. On the other hand, the second strategy can only
search for feasible LVQ solutions within a constrained region,
which prevents the full potential of LVQ from being exploited.
In some cases, the learned linear transform may degenerate
into an identity transform when all diagonal elements approach
one, leading to limited improvements in R-D performance.
To overcome the above two shortcomings and enable LVQ
search in a less constrained space, we propose a novel method
to optimize SALVQ more effectively. Specifically, we param-
eterize the lattice basis matrix B using its singular value
decomposition (SVD):
B = UΣVT
(2)
where U and V are learnable orthogonal linear transforms,
and Σ is a learnable diagonal matrix with non-zero entries
to ensure invertibility. We enforce the orthogonality of U
and V using the orthogonal parametrization of PyTorch [51],
allowing flexible and stable optimization of B. This SVD-
based parameterization enables flexible and diverse lattice
bases to be learned during training. In some special cases
where all singular values approach one, the lattice basis matrix
degenerates to UVT, which acts as a rotation matrix, resulting
in the lattice Voronoi cell becoming a rotated hypercube in
high-dimensional space. Notably, a suitably rotated hypercube
can better capture correlations within feature vectors than
the canonical hypercube that corresponds to the Voronoi cell
of USQ, thereby reducing quantization error along principal
directions. In more general cases, this parameterization enables
a rich set of feasible lattice basis matrix to be explored and
optimized jointly with the 3DGS model. As a result, the system
automatically discovers the SALVQ configuration that best
suits the data, leading to improved R-D performance with
negligible computational overhead.
C. Implementation details
In anchor-based 3DGS compression, three types of anchor
attributes require quantization: the latent features f, the scaling
factors l, and the K learnable offsets {Oi}K
i=1. We apply the
proposed SALVQ approach only to the latent features f, as
they account for the majority of the total bit budget, mak-
ing their efficient compression particularly critical. Moreover,
since the latent features f are high-dimensional, applying LVQ
to them typically yields greater gains compared to lower-
dimensional attributes. Because the learned lattice basis B
is unconstrained, it will generally not coincide with highly
structured lattices (e.g., Dn, E8) that admit coset representa-
tions for fast exact quantization [44]. To avoid computationally
intensive nearest-lattice-point search under an arbitrary basis,

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
we follow Zhang et al. [41] and adopt Babai’s rounding
technique (BRT) [52]:
ql(y) = B⌊B−1y⌉
(3)
where ⌊·⌉represents the rounding operation. BRT provides
an efficient approximation to nearest-lattice-point assignment
under the basis B, enabling fast quantization in practice.
Before feeding the latent features f into the proposed
SALVQ, we first perform mean removal to centralize the
features. The shift vector used in this mean removal can be
either spatially adaptive mean vectors µ predicted by the
hyperprior, or a global mean vector µg. When integrating
SALVQ into a hyperprior-only architecture such as HAC [14],
we adopt the first option. The centralized latent features,
fc
= f −µ, are quantized using BRT. Specifically, we
transform fc into ft = B−1fc using the learned lattice basis
matrix B. Subsequently, we quantize ft into ˆft through a
rounding operation, which is approximated by adding uniform
noise during training and replaced by hard quantization during
inference. The entropy coding of ˆft uses adaptive arithmetic
coding, with the probability model for ˆft defined as:
p(ˆft) =
Y
i
h
N(0, σ2
i ) ∗U(−qs
2 , qs
2 )

( ˆf (i)
t )
i
(4)
where qs is the quantization step size. In this framework, the
hyperprior estimates spatially adaptive mean vectors for mean
removal and predicts the variances of a zero-mean Gaussian
distribution assumed for the quantized representation. In the
decompression stage, the ˆft is first decoded and then recovered
to ˆfc = Bˆft. Finally, the latent feature is reconstructed by
adding the mean vectors µ predicted by the hyperprior, i.e,
ˆf = ˆfc + µ.
When integrating the proposed SALVQ approach into archi-
tectures with complex context models such as ContextGS [18],
the mean vectors µ for different groups are typically estimated
sequentially. To avoid sequential quantization, we adopt a
one-pass quantization strategy, ensuring that quantization is
completed before entropy coding. To enable this one-pass
quantization, we use a global mean vector µg for mean
removal. In this case, the assumption that ˆft follows a zero-
mean Gaussian distribution becomes invalid, and the entropy
models for ˆft remain the same as in the original architectures.
Specifically, the entropy model for ˆft is a Gaussian distribution
with a non-zero mean, and the parameters of these probability
distributions are predicted using both the hyperprior and the
causal context model.
D. Rate control scheme
By replacing the USQ module in existing anchor-based
3DGS compression architectures with the proposed SALVQ,
we enhance the system to achieve comparable or higher
rendering quality with a smaller memory footprint. However,
this system can only operate at a fixed rate. This is because
the R-D trade-off is controlled by the Lagrange multiplier λ
in the loss function
L = Ldistortion + λLrate + λregLreg
(5)
where Ldistortion denotes the rendering distortion, Lrate corre-
sponds to the rate, Lreg represents other regularization terms
controlled by λreg. For a specific λ, the 3DGS model is opti-
mized to achieve the corresponding rate target. For each choice
of λ, the 3DGS model is trained to target a specific rate, and
once training is complete, the model operates only at that rate.
Supporting multiple R-D trade-offs requires training separate
models with different λ values, leading to computational and
memory costs that grow linearly with the number of desired
rates. This inefficiency highlights the importance of developing
variable-rate compression methods that enable a single model
to support multiple R-D trade-offs.
Understanding how the Lagrange multiplier λ affects the R-
D trade-off is essential for enabling variable-rate compression.
A higher λ penalizes the rate term more, reducing the rate at
the cost of higher distortion. This occurs because a stronger
penalty forces the anchor attributes to adopt a lower dynamic
range, reducing the number of lattice points used. Decreasing
λ has the opposite effect, allowing a larger dynamic range
and lower distortion. This mirrors neural image compression,
where varying the quantization step size provides variable-rate
control [42], [53]–[56]; we adopt the same idea for 3DGS. Mo-
tivated by this, we fix the anchor attributes’ dynamic range and
learn rate-specific lattice densities to control the R–D trade-off.
Specifically, we introduce a gain vector g = [g1, · · · , gM]1,
where M is the number of target rates, to control the lattice
density for each rate. The anchor attributes and the base
quantization step qs are shared across all targets; target i uses
its gain gi to scale the quantization step from qs to giqs.
A larger gain leads to a coarser LVQ and a lower bitrate,
enabling flexible rate control without retraining the model. To
learn the gain vector, we associate each target with a Lagrange
multiplier λi. At each training iteration, we randomly sample
an index from {1, · · · , M}, apply gi to modulate the step, and
use λi in the loss. After training, the anchor attributes and qs
remain fixed; at inference one simply selects a learned gain gi
to meet the desired bitrate.
This flexible rate control scheme is applicable to both
SALVQ and USQ, noting that USQ can be considered as a
special case of SALVQ where the identity matrix is used as the
lattice basis matrix. When using this rate control scheme with
either SALVQ or USQ, the distortion at each target rate is pri-
marily determined by the quantization error. Under comparable
bitrates, SALVQ typically achieves lower quantization error
compared to USQ, resulting in lower distortion. Therefore,
combining this rate control strategy with our proposed SALVQ
leads to better overall performance.
V. EXPERIMENT
In this section, we describe the experimental setup; present
the gains of SALVQ over USQ along with its memory and
compute overhead; compare SALVQ-based 3DGS compres-
sion methods to earlier 3DGS compression baselines; assess
visual quality differences between USQ- and SALVQ-based
methods; evaluate both quantizers under variable-rate and
1The gain vector is learned separately for anchor latent features, scaling
factors, and offsets.

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
TABLE I: BD-rate of LVQ relative to USQ on HAC [14],
HAC++ [17] and ContextGS [18] across three datasets. Lower
is better; negative denotes improved compression efficiency.
Bold numbers highlight the best results.
HAC [14]
HAC++ [17]
ContextGS [18]
E8 LVQ
SALVQ
E8 LVQ SALVQ
E8 LVQ SALVQ
Mip-NeRF360 [58]
-2.04%
-13.48%
-1.52%
-4.55%
-0.63%
-5.71%
Tank&Temples [59]
-4.42%
-16.16%
-2.35%
-8.95%
-2.12%
-8.69%
DeepBlending [60]
-8.12%
-13.44%
-2.78%
-7.75%
-0.93%
-9.75%
progressive compression; and conclude with limitations and
directions for future work.
A. Experimental setting
1) Baselines: Our experiments are mainly based on three
baselines: the earlier HAC [14], the more recent HAC++ [17]
and ContextGS [18]. We also assess SALVQ on PCGS [57],
a progressive 3DGS compression baseline. To evaluate the
effectiveness of the proposed SALVQ approach, we replace the
USQ module in these architectures with the proposed SALVQ,
while keeping all other components unchanged.
2) Dataset:
We
evaluate
the
R-D
performance
on
three commonly used large-scale real-scene datasets: Mip-
NeRF360 [58], Tanks&Temples [59], and DeepBlending [60].
Notably, we assess all nine scenes from the Mip-NeRF360
dataset [58]. These diverse datasets provide a comprehensive
evaluation of the proposed SALVQ approach.
3) Distortion metrics: Following the evaluation settings
commonly used in previous works, we evaluate rendering
distortion using PSNR, SSIM [61], and LPIPS [62].
4) Rate metrics: We use the size of the encoded bitstream
(in MB) as the rate metric. To quantitatively evaluate the
improvements introduced by the proposed SALVQ approach,
we adopt the BD-rate [63] to measure the average reduction
in memory footprint at a fixed distortion level. A negative
BD-rate indicates that the compression system achieves a
smaller memory footprint at the same quality compared to
the baseline, with its absolute value reflecting the magnitude
of the performance gain.
5) Training details: Following the official settings, we
train
each
scene
for
30k
iterations
for
HAC
[14],
HAC++ [17], and ContextGS [18], and for 40k iterations for
PCGS [57]. The Lagrange multiplier λ is varied over the set
{0.002, 0.004, 0.008, 0.015, 0.025} to evaluate compression
performance across a wider range. For other hyperparameters,
we follow the same settings as HAC [14], HAC++ [17],
ContextGS [18] and PCGS [57].
B. Performance gain from SALVQ over USQ
We conducted comprehensive experiments to evaluate the
effectiveness of our proposed SALVQ approach. Specifically,
we integrated the proposed SALVQ into three existing ar-
chitectures: HAC [14], HAC++ [17], and ContextGS [18].
These three baselines cover different design choices in the
use of context models, including hyperprior-only, channel-wise
autoregressive, and checkerboard-like spatial autoregressive
models, respectively. For reproducibility, we evaluate only
baselines with stable, publicly available code. Accordingly, we
exclude HEMGS [16] (no released code) and CAT-3DGS [15]
(issues in the current public release that prevent a reliable
evaluation). Notably, both CAT-3DGS and HAC++ employ
channel-wise autoregressive entropy models; our extensive
results on HAC++ therefore already characterize SALVQ’s
behavior for this design family. We will include these addi-
tional baselines once stable, accessible implementations are
available.
As shown in Fig. 3, the proposed SALVQ consistently
improves the R-D performance across all architectures on all
three datasets. Tab. I quantifies these benefits: at the same
distortion level, integrating SALVQ helps reduce the memory
footprint, with average savings ranging from 4.55% in the
worst case to 16.16% in the best. To highlight the benefit
of scene-adaptive LVQ over a fixed lattice quantizer (such as
one built with the E8 lattice), we form a reference baseline
by reducing the latent dimension from 50 to 48 and replacing
USQ with a product E8 lattice quantizer. As shown in Fig. 3
and Tab. I, our SALVQ consistently outperforms this fixed-
lattice baseline. This is primarily because a fixed lattice quan-
tizer is designed under the assumption of a uniform, scene-
invariant latent distribution, which often does not align with
the actual distribution of the latent features in each specific
scene. Unlike generalizable compression, 3DGS compression
is scene-specific; learning the lattice basis per scene can
better align the quantizer with the true source statistics and
significantly improve R-D performance.
Moreover, Fig. 3 and Tab. I show that pairing SALVQ
with a less powerful entropy model such as HAC [14]
yields larger gains than doing so with stronger models like
HAC++ [17] and ContextGS [18]. This is expected: when the
entropy model has limited capacity to capture correlations,
a more sophisticated quantizer can compensate by capturing
correlation in the quantization stage and improving space-
filling efficiency. Consequently, SALVQ is well suited for
compression models without autoregressive priors, offering
strong R-D performance while keeping coding latency low.
We report per-scene results of pairing our SALVQ ap-
proach with existing 3DGS compression architectures, eval-
uated across multiple fidelity metrics (PSNR, SSIM [61], and
LPIPS [62]) on three datasets. Detailed results are provided in
the supplementary material.
C. Memory and computational overhead
Beyond the BD-rate gains, the cost of achieving the R–D
gains is pivotal in persuading developers to adopt SALVQ
rather than USQ. In this subsection, we evaluate the memory
and compute costs introduced by SALVQ.
Since SALVQ is applied only to the anchor latent feature
with a dimension of 50, it introduces just 502 + 502 + 50
additional parameters to parameterize the adaptive lattice basis
matrix B (see Eq. (2)). Storing these parameters in float32
adds only 0.02 MB of memory, which is negligible relative to
the memory footprint of bitstream. Tab. II reports the training,
encoding, and decoding times across various architectures,
comparing the proposed SALVQ with USQ. Since rendering

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
8
10
12
14
16
18
20
Size/MB
26.4
26.6
26.8
27
27.2
27.4
27.6
27.8
PSNR/dB
HAC (Ours SALVQ)
HAC (E8 LVQ)
HAC (USQ)
(a) HAC on Mip-NeRF360
4
5
6
7
8
9
10
Size/MB
23.5
23.6
23.7
23.8
23.9
24
24.1
24.2
24.3
24.4
PSNR/dB
HAC (Ours SALVQ)
HAC (E8 LVQ)
HAC (USQ)
(b) HAC on Tank&Temples
2.5
3
3.5
4
4.5
5
5.5
Size/MB
28.8
29
29.2
29.4
29.6
29.8
30
30.2
30.4
PSNR/dB
HAC (Ours SALVQ)
HAC (E8 LVQ)
HAC (USQ)
(c) HAC on DeepBlending
6
8
10
12
14
16
Size/MB
26.9
27
27.1
27.2
27.3
27.4
27.5
27.6
27.7
27.8
PSNR/dB
ContextGS (Ours SALVQ)
ContextGS (E8 LVQ)
ContextGS (USQ)
(d) ContextGS on Mip-NeRF360
3
4
5
6
7
8
9
Size/MB
23.7
23.8
23.9
24
24.1
24.2
24.3
24.4
24.5
PSNR/dB
ContextGS (Ours SALVQ)
ContextGS (E8 LVQ)
ContextGS (USQ)
(e) ContextGS on Tank&Temples
1.5
2
2.5
3
3.5
4
4.5
Size/MB
29.3
29.4
29.5
29.6
29.7
29.8
29.9
30
30.1
30.2
30.3
PSNR/dB
ContextGS (Ours SALVQ)
ContextGS (E8 LVQ)
ContextGS (USQ)
(f) ContextGS on DeepBlending
3
4
5
6
7
8
9
10
11
12
Size/MB
26.8
26.9
27
27.1
27.2
27.3
27.4
27.5
27.6
27.7
27.8
PSNR/dB
HAC++ (Ours SALVQ)
HAC++ (E8 LVQ)
HAC++ (USQ)
(g) HAC++ on Mip-NeRF360
2
3
4
5
6
7
Size/MB
23.6
23.7
23.8
23.9
24
24.1
24.2
24.3
24.4
PSNR/dB
HAC++ (Ours SALVQ)
HAC++ (E8 LVQ)
HAC++ (USQ)
(h) HAC++ on Tank&Temples
1
1.5
2
2.5
3
3.5
4
4.5
Size/MB
29.2
29.4
29.6
29.8
30
30.2
30.4
PSNR/dB
HAC++ (Ours SALVQ)
HAC++ (E8 LVQ)
HAC++ (USQ)
(i) HAC++ on DeepBlending
Fig. 3: R–D curves of HAC [14], ContextGS [18] and HAC++ [17] under different quantizers.
occurs after the decompression of anchor attributes and all
architectures employ the same rendering pipeline as Scaffold-
GS [13], the choice of quantizer does not affect rendering time.
Therefore, rendering-speed comparisons are omitted. Tab. II
shows that replacing USQ with the proposed SALVQ results in
a slight increase in training time while maintaining comparable
encoding and decoding times. While training takes slightly
longer, it is handled by the 3DGS asset producer and does
not affect the end-user experience. Given the already long
per-scene training time, adding a few more minutes is not
a practical concern.
Overall, the proposed SALVQ approach is low-complexity
and cost-effective: it delivers significant R-D performance
improvements with little additional computation or parameters,
making it a strong candidate for integration as a basic module
in 3DGS compression systems.
D. Comparison with other 3DGS compression methods
After completing comparisons among anchor-based com-
pression methods, we further present a comprehensive eval-
uation against other existing methods, including commonly
used baselines [6]–[13], [19], [64], [65] and the recent feed-
forward compression method FCGS [66]. Since these meth-
ods typically operate at only one or two bitrate targets, we
cannot directly compare with them using R-D curves and
BD-rate calculations. Therefore, we use our model trained
with λ = 0.004 for evaluation, and report the results in
Tab. III. As shown in Tab. III, combining our SALVQ approach
with any anchor-based compression architecture [14], [17],
[18] consistently outperforms these baselines in compression
performance. On commonly used datasets, including Mip-
NeRF 360 [58], Tanks and Temples [59], and DeepBlend-
ing [60], integrating our SALVQ approach with the current

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
USQ
Ours LVQ
GT
USQ
Ours LVQ
2.23 MB/31.25 dB
2.20 MB/31.41 dB
13.27 MB/22.04 dB
12.66 MB/22.73 dB
11.65 MB/24.05 dB
11.60 MB/32.26 dB
5.04 MB/18.64 dB
4.88 MB/24.43 dB
Fig. 4: Visual comparison of different quantizers on four scenes: ‘playroom’ (DeepBlending [60]), ‘flower’ and ‘stump’ (Mip-
NeRF360 [58]), and ‘train’ (Tanks and Temples [59]), shown from the first to the fourth row, respectively. The last row
shows the Y-channel residual maps for both quantization methods, providing a more intuitive visualization of the differences
in brightness preservation on the ‘train’ scene.

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
TABLE II: Comparison of computational cost between USQ and SALVQ. The proposed SALVQ approach incurs a slight
increase in training time but has negligible impact on encoding and decoding times.
Training time
Encoding time
Decoding time
HAC [14]
HAC++ [17]
ContextGS [18]
HAC [14]
HAC++ [17]
ContextGS [18]
HAC [14]
HAC++ [17]
ContextGS [18]
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
USQ
SALVQ
Mip-NeRF360 [58]
1949
2206
2735
2937
3927
4167
4.20
4.56
8.48
8.37
32.73
31.93
10.05
10.76
13.86
13.81
33.32
31.84
Tank&Temples [59]
1385
1701
1880
2055
2505
2639
2.51
2.80
5.63
5.30
29.37
31.64
5.77
6.43
8.81
8.61
29.05
28.80
DeepBlending [60]
1533
1766
1857
2060
2320
2375
1.33
1.49
2.98
2.91
16.78
12.59
2.85
3.11
4.41
4.42
16.63
12.49
TABLE III: Comparison between SALVQ-based compression systems and other 3DGS compression methods, including 3DGS
and Scaffold-GS, for reference. Bold numbers and underline highlight the best and the second-best results, respectively. The
size values are measured in megabytes (MB).
Datasets
Mip-NeRF360 [58]
Tank&Temples [59]
DeepBlending [60]
Methods
psnr↑
ssim↑
lpips↓
size↓
psnr↑
ssim↑
lpips↓
size↓
psnr↑
ssim↑
lpips↓
size↓
3DGS [6]
27.46
0.812
0.222
750.9
23.69
0.844
0.178
431.0
29.42
0.899
0.247
663.9
Scaffold-GS [13]
27.50
0.806
0.252
253.9
23.96
0.853
0.177
86.50
30.21
0.906
0.254
66.00
Compact3DGS [8]
27.08
0.798
0.247
48.80
23.32
0.831
0.201
39.43
29.79
0.901
0.258
43.21
Compressed3D [12]
26.98
0.801
0.238
28.80
23.32
0.832
0.194
17.28
29.38
0.898
0.253
25.30
EAGLES [19]
27.14
0.809
0.231
58.91
23.28
0.835
0.203
28.99
29.72
0.906
0.249
52.34
LightGaussian [9]
27.00
0.799
0.249
44.54
22.83
0.822
0.242
22.43
27.01
0.872
0.308
33.94
SOG et al. [64]
26.56
0.791
0.241
16.70
23.15
0.828
0.198
9.30
29.12
0.892
0.270
5.70
Navaneet et al. [11]
27.12
0.806
0.240
19.33
23.44
0.838
0.198
12.50
29.90
0.907
0.251
13.50
Reduced3DGS [7]
27.19
0.807
0.230
29.54
23.57
0.840
0.188
14.00
29.63
0.902
0.249
18.00
RDOGaussian [10]
27.05
0.802
0.239
23.46
23.34
0.835
0.195
12.03
29.63
0.902
0.252
18.00
CompGS [65]
27.26
0.803
0.239
16.50
23.70
0.837
0.208
9.60
29.69
0.901
0.279
8.77
FCGS [66]
27.05
0.798
0.237
36.30
23.48
0.833
0.193
18.80
29.27
0.893
0.257
30.10
HAC [14] + Ours SALVQ
27.60
0.807
0.239
14.30
24.20
0.847
0.186
7.27
30.06
0.903
0.268
3.98
ContextGS [18] + Ours SALVQ
27.62
0.808
0.238
12.12
24.35
0.852
0.184
6.81
30.14
0.907
0.266
3.30
HAC++ [17] + Ours SALVQ
27.61
0.803
0.252
8.31
24.26
0.849
0.190
5.22
30.18
0.907
0.266
2.87
SOTA HAC++ [17] achieves an average compression ratio
of 131× over the original 3DGS models and 23.7× over
Scaffold-GS, while delivering higher or comparable rendering
quality. As new SOTA methods continue to emerge, combining
them with our SALVQ approach is expected to yield even
higher compression ratios without compromising rendering
quality.
E. Visual comparison
We provide visual comparisons between our SALVQ ap-
proach and USQ on the ContextGS baseline [18] in Fig. 4.
The only difference between the two models is the quantizer
used, and both are trained with a Lagrange multiplier of
λ = 0.008. We report the bitstream size (in MB) and the
PSNR (in dB) of the corresponding patches for each case. To
demonstrate the effectiveness of the proposed LVQ in reducing
rendering distortion, we evaluate four representative scenes:
‘playroom’ (DeepBlending [60]), ‘flower’ and ‘stump’ (Mip-
NeRF360 [58]), and ‘train’ (Tanks and Temples [59]), shown
from the first to the fourth row in Fig. 4. In the ‘playroom’
scene, our method better preserves the structure of the wall-
mounted switch, which appears appears as an unrecognizable
blur in the USQ result. In the ‘flower’ scene, it recovers small
flower details that are missing with USQ. For the ‘stump’
scene, our method effectively avoids floater artifacts. In the
‘train’ scene, it better preserves the correct brightness, whereas
USQ results are overly bright. The difference in brightness
preservation is clearly visible in the residual maps of the Y-
channel provided in the last row of Fig. 4.
F. Evaluating rate adaptation capability
After evaluating single-rate compression with separately
trained models for each target rate, we turn to variable-rate
compression, where a single model supports multiple rates.
Building on the rate-control scheme in Sec.IV-D, we compare
LVQ-based and USQ-based variable-rate methods through
extensive experiments. Because this rate-control scheme strug-
gles to cover a wide range of memory budgets, we restrict our
evaluation to the high-rate region of the R–D curves in Fig. 3.
We set the Lagrange multiplier to λ ∈{0.002, 0.004, 0.008};
an additional value λ = 0.006 is included to provide the four
points required for BD-rate computation.
As shown in Fig.5, the proposed SALVQ approach consis-
tently outperforms USQ in variable-rate mode. Table IV re-
ports the BD-rate, which quantifies the impact of converting a
USQ single-rate model to variable-rate mode. This conversion
typically degrades performance for USQ-based variable-rate
models (a positive BD-rate). In contrast, SALVQ in variable-
rate mode often matches or surpasses the R–D performance
of USQ single-rate models; when it does not, it markedly
reduces the performance loss relative to USQ-based variable-
rate systems. Overall, with four rate targets, our LVQ-based
variable-rate models deliver competitive R-D performance
while achieving a 4× reduction in total training time and
model memory footprint compared with training separate USQ
single-rate compression models.
To account for these gains, we examine the dominant distor-
tion source in variable-rate operation. Because rate control is
implemented by adjusting the quantization step size, distortion

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
11
12
13
14
15
16
17
18
19
Size/MB
27.1
27.2
27.3
27.4
27.5
27.6
27.7
27.8
PSNR/dB
HAC (USQ VBR)
HAC (USQ)
HAC (SALVQ VBR)
(a) HAC on Mip-NeRF360
6
6.5
7
7.5
8
8.5
9
9.5
10
Size/MB
23.95
24
24.05
24.1
24.15
24.2
24.25
24.3
24.35
PSNR/dB
HAC (USQ VBR)
HAC (USQ)
HAC (SALVQ VBR)
(b) HAC on Tank&Temples
3.5
4
4.5
5
5.5
Size/MB
29.4
29.5
29.6
29.7
29.8
29.9
30
30.1
30.2
30.3
PSNR/dB
HAC (USQ VBR)
HAC (USQ)
HAC (SALVQ VBR)
(c) HAC on DeepBlending
9
10
11
12
13
14
15
16
Size/MB
27.3
27.35
27.4
27.45
27.5
27.55
27.6
27.65
27.7
PSNR/dB
ContextGS (USQ VBR)
ContextGS (USQ)
ContextGS (SALVQ VBR)
(d) ContextGS on Mip-NeRF360
5.5
6
6.5
7
7.5
8
8.5
Size/MB
24
24.05
24.1
24.15
24.2
24.25
24.3
24.35
24.4
24.45
PSNR/dB
ContextGS (USQ VBR)
ContextGS (USQ)
ContextGS (SALVQ VBR)
(e) ContextGS on Tank&Temples
2.5
3
3.5
4
4.5
Size/MB
29.8
29.85
29.9
29.95
30
30.05
30.1
30.15
30.2
30.25
30.3
PSNR/dB
ContextGS (USQ VBR)
ContextGS (USQ)
ContextGS (SALVQ VBR)
(f) ContextGS on DeepBlending
6
7
8
9
10
11
12
Size/MB
27.35
27.4
27.45
27.5
27.55
27.6
27.65
27.7
27.75
PSNR/dB
HAC++ (USQ VBR)
HAC++ (USQ)
HAC++ (SALVQ VBR)
(g) HAC++ on Mip-NeRF360
3.5
4
4.5
5
5.5
6
6.5
7
Size/MB
24
24.05
24.1
24.15
24.2
24.25
24.3
24.35
PSNR/dB
HAC++ (USQ VBR)
HAC++ (USQ)
HAC++ (SALVQ VBR)
(h) HAC++ on Tank&Temples
2
2.5
3
3.5
4
Size/MB
29.9
29.95
30
30.05
30.1
30.15
30.2
30.25
30.3
30.35
30.4
PSNR/dB
HAC++ (USQ VBR)
HAC++ (USQ)
HAC++ (SALVQ VBR)
(i) HAC++ on DeepBlending
Fig. 5: Single-rate vs. variable-rate compression performance on three architectures and datasets. Here, ‘VBR’ indicates that
the R-D curve corresponds to a variable rate compression method.
5
6
7
8
9
10
11
12
13
14
Size/MB
27.1
27.2
27.3
27.4
27.5
27.6
27.7
PSNR/dB
PCGS (Ours SALVQ)
PCGS (USQ)
(a) Mip-NeRF360 [58]
3
4
5
6
7
8
9
10
Size/MB
24.1
24.15
24.2
24.25
24.3
24.35
24.4
24.45
PSNR/dB
PCGS (Ours SALVQ)
PCGS (USQ)
(b) Tank&Temples [59]
2
2.5
3
3.5
4
4.5
5
Size/MB
29.65
29.7
29.75
29.8
29.85
29.9
29.95
30
30.05
30.1
30.15
PSNR/dB
PCGS (Ours SALVQ)
PCGS (USQ)
(c) DeepBlending [60]
Fig. 6: Comparison of two quantizers in the PCGS [57] architecture

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
TABLE IV: BD-rate of SALVQ-based and USQ-based variable-rate systems relative to that architecture’s USQ-based single-
rate baseline. Bold numbers highlight cases where variable-rate compression models outperform the corresponding USQ-based
single-rate counterparts in R-D performance.
HAC [14]
ContextGS [18]
HAC++ [17]
USQ-VBR
SALVQ-VBR
USQ-VBR
SALVQ-VBR
USQ-VBR
SALVQ-VBR
Mip-NeRF360 [58]
4.54%
-6.44%
2.49%
-5.01%
14.99%
4.16%
Tank&Temples [59]
-0.99%
-13.83%
9.87%
-4.65%
18.89%
4.62%
DeepBlending [60]
1.74%
-16.89%
1.94%
-18.04%
4.94%
-8.30%
is dominated by quantization error. At comparable bitrates, the
proposed SALVQ approach yields smaller quantization errors
than USQ, resulting in lower rendering distortion and a better
R-D trade-off. This makes SALVQ particularly advantageous
in variable-rate mode.
G. Evaluating SALVQ in progressive compression
Besides
the
single-rate
and
variable-rate
compression
modes, we also evaluate the progressive coding mode of
SALVQ for 3DGS compression. In this setting, partial bit-
streams are firstly decoded into a coarse 3DGS representation
by which a low visual quality scene can be rendered, and
the quality improves as additional bits are further decoded.
PCGS [57] is representative of the SOTA in progressive 3DGS
compression. We replace the USQ used in PCGS with the
proposed SALVQ and assess its impact on R-D performance.
As shown in Fig. 6, replacing USQ with SALVQ yields
consistent gains: the BD-rate gains of SALVQ-based PCGS
over USQ-based PCGS on the three datasets (left to right) are
−3.47%, −20.57%, and −7.01%, respectively.
H. Limitation discussion
In consideration of high training cost, it is difficult to
realize a high granularity of rate control. Given computation
resources and the limit of training time, if the number of
discrete target rates is large, the effective training time per
rate is reduced, which can degrade R–D performance. In our
implementation, the variable-rate mode of SALVQ only offers
a modest adjustable rate range, with ratio of the highest rate
over the lowest rate being around 1.5.
VI. CONCLUSION
This study investigates the use of LVQ in 3DGS com-
pression. Specifically, we propose a novel SALVQ method,
which learns a scene-adaptive lattice basis, making the quan-
tizer’s Voronoi cell geometry learnable and aligned to scene
statistics. SALVQ can be seamlessly integrated into existing
3DGS compression pipelines to improve R-D performance
without requiring any modification to other components. It
also maintains a computational cost comparable to that of
USQ. Furthermore, by adaptively scaling the lattice basis
vectors, SALVQ provides effective variable-rate control, of-
ten matching or surpassing the R–D performance of USQ-
based single-rate models, while eliminating the need to train
separate models for different R–D targets. This significantly
reduces both training time and memory footprint. Extensive
results across multiple architectures show that replacing the
USQ module with SALVQ yields consistent gains, positioning
SALVQ as a modular, broadly applicable building block for
3DGS compression.
REFERENCES
[1] H. Xu, “Prior-guided neural compression of visual data,” Ph.D. thesis,
McMaster University, Hamilton, Canada, 2025. [Online]. Available:
http://hdl.handle.net/11375/32258
[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[3] S. Lombardi, T. Simon, J. Saragih, G. Schwartz, A. Lehrmann, and
Y. Sheikh, “Neural volumes: Learning dynamic renderable volumes from
images,” arXiv preprint arXiv:1906.07751, 2019.
[4] V. Sitzmann, M. Zollh¨ofer, and G. Wetzstein, “Scene representation
networks: Continuous 3d-structure-aware neural scene representations,”
Advances in neural information processing systems, vol. 32, 2019.
[5] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ra-
mamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view
synthesis with prescriptive sampling guidelines,” ACM Transactions on
Graphics (ToG), vol. 38, no. 4, pp. 1–14, 2019.
[6] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[7] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis,
“Reducing the memory footprint of 3d gaussian splatting,” Proceedings
of the ACM on Computer Graphics and Interactive Techniques, vol. 7,
no. 1, pp. 1–17, 2024.
[8] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 719–21 728.
[9] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
arXiv preprint arXiv:2311.17245, 2023.
[10] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, “End-to-
end rate-distortion optimized 3d gaussian representation,” in European
Conference on Computer Vision.
Springer, 2024, pp. 76–92.
[11] K. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and
H. Pirsiavash, “Compgs: Smaller and faster gaussian splatting with
vector quantization,” in European Conference on Computer Vision.
Springer, 2024, pp. 330–349.
[12] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2024, pp. 10 349–10 358.
[13] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[14] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid
assisted context for 3d gaussian splatting compression,” in European
Conference on Computer Vision.
Springer, 2024, pp. 422–438.
[15] Y.-T. Zhan, C.-Y. Ho, H. Yang, Y.-H. Chen, J. C. Chiang, Y.-L. Liu,
and W.-H. Peng, “CAT-3DGS: A context-adaptive triplane approach
to rate-distortion-optimized 3DGS compression,” in The Thirteenth
International Conference on Learning Representations, 2025. [Online].
Available: https://openreview.net/forum?id=m3KuuE2ozw
[16] L. Liu, Z. Chen, and D. Xu, “Hemgs: A hybrid entropy model for 3d
gaussian splatting data compression,” arXiv preprint arXiv:2411.18473,
2024.

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
[17] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac++: Towards 100x
compression of 3d gaussian splatting,” IEEE Transactions on Pattern
Analysis and Machine Intelligence, pp. 1–17, 2025.
[18] Y. Wang, Z. Li, L. Guo, W. Yang, A. Kot, and B. Wen, “Contextgs: Com-
pact 3d gaussian splatting with anchor level context model,” Advances
in neural information processing systems, vol. 37, pp. 51 532–51 551,
2024.
[19] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated
3d gaussians with lightweight encodings,” in European Conference on
Computer Vision.
Springer, 2024, pp. 54–71.
[20] S. Xie, W. Zhang, C. Tang, Y. Bai, R. Lu, S. Ge, and Z. Wang, “Mesongs:
Post-training compression of 3d gaussians via efficient attribute trans-
formation,” in European Conference on Computer Vision.
Springer,
2024, pp. 434–452.
[21] M. Niemeyer, F. Manhardt, M.-J. Rakotosaona, M. Oechsle, D. Duck-
worth, R. Gosula, K. Tateno, J. Bates, D. Kaeser, and F. Tombari,
“Radsplat: Radiance field-informed gaussian splatting for robust real-
time rendering with 900+ fps,” arXiv preprint arXiv:2403.13806, 2024.
[22] M. S. Ali, S.-H. Bae, and E. Tartaglione, “Elmgs: Enhancing memory
and computation scalability through compression for 3d gaussian splat-
ting,” arXiv preprint arXiv:2410.23213, 2024.
[23] A. Hanson, A. Tu, V. Singla, M. Jayawardhana, M. Zwicker, and
T. Goldstein, “Pup 3d-gs: Principled uncertainty pruning for 3d gaussian
splatting,” arXiv preprint arXiv:2406.10219, 2024.
[24] W. Liu, T. Guan, B. Zhu, L. Ju, Z. Song, D. Li, Y. Wang, and
W. Yang, “Efficientgs: Streamlining gaussian splatting for large-scale
high-resolution scene representation,” arXiv preprint arXiv:2404.12777,
2024.
[25] Y. Lee, Z. Zhang, and D. Fan, “Safeguardgs: 3d gaussian primitive
pruning while avoiding catastrophic scene destruction,” arXiv preprint
arXiv:2405.17793, 2024.
[26] G. Fang and B. Wang, “Mini-splatting: Representing scenes with a
constrained number of gaussians,” in European Conference on Computer
Vision.
Springer, 2024, pp. 165–181.
[27] C. Zhang, D. Florencio, and C. Loop, “Point cloud attribute compression
with graph transform,” in 2014 IEEE International Conference on Image
Processing (ICIP).
IEEE, 2014, pp. 2066–2070.
[28] R. L. De Queiroz and P. A. Chou, “Compression of 3d point clouds
using a region-adaptive hierarchical transform,” IEEE Transactions on
Image Processing, vol. 25, no. 8, pp. 3947–3956, 2016.
[29] S. Gu, J. Hou, H. Zeng, H. Yuan, and K.-K. Ma, “3d point cloud
attribute compression using geometry-guided sparse representation,”
IEEE Transactions on Image Processing, vol. 29, pp. 796–808, 2019.
[30] P. A. Chou, M. Koroteev, and M. Krivoku´ca, “A volumetric approach to
point cloud compression—part i: Attribute compression,” IEEE Trans-
actions on Image Processing, vol. 29, pp. 2203–2216, 2019.
[31] X. Sheng, L. Li, D. Liu, and Z. Xiong, “Attribute artifacts removal for
geometry-based point cloud compression,” IEEE Transactions on Image
Processing, vol. 31, pp. 3399–3413, 2022.
[32] J. Wang and Z. Ma, “Sparse tensor-based point cloud attribute com-
pression,” in 2022 IEEE 5th International Conference on Multimedia
Information Processing and Retrieval (MIPR).
IEEE, 2022, pp. 59–64.
[33] Y. He, X. Ren, D. Tang, Y. Zhang, X. Xue, and Y. Fu, “Density-
preserving deep point cloud compression,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 2333–2342.
[34] H. Xu, X. Zhang, and X. Wu, “Fast point cloud geometry compression
with context-based residual coding and inr-based refinement,” in Euro-
pean Conference on Computer Vision.
Springer, 2024, pp. 270–288.
[35] D. Minnen and S. Singh, “Channel-wise autoregressive entropy models
for learned image compression,” in 2020 IEEE International Conference
on Image Processing (ICIP).
IEEE, 2020, pp. 3339–3343.
[36] D. He, Y. Zheng, B. Sun, Y. Wang, and H. Qin, “Checkerboard context
model for efficient learned image compression,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 14 771–14 780.
[37] D. Minnen, J. Ball´e, and G. D. Toderici, “Joint autoregressive and
hierarchical priors for learned image compression,” Advances in neural
information processing systems, vol. 31, 2018.
[38] X. Zhang and X. Wu, “Lvqac: Lattice vector quantization coupled with
spatially adaptive companding for efficient learned image compression,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 10 239–10 248.
[39] E. Lei, H. Hassani, and S. S. Bidokhti, “Approaching rate-distortion
limits in neural compression with lattice transform coding,” in The
Thirteenth International Conference on Learning Representations, 2025.
[Online]. Available: https://openreview.net/forum?id=Tv36j85SqR
[40] M. Cao, W. Dai, S. Li, H. Li, C. Li, J. Zou, and H. Xiong, “Entropy
relaxed lattice vector quantization for learned image compression,” in
2024 Data Compression Conference (DCC). IEEE, 2024, pp. 548–548.
[41] X. Zhang and X. Wu, “Learning optimal lattice vector quantizers for
end-to-end neural image compression,” Advances in Neural Information
Processing Systems, vol. 37, pp. 106 497–106 518, 2024.
[42] H. Xu, X. Wu, and X. Zhang, “Multirate neural image compression with
adaptive lattice vector quantization,” in Proceedings of the Computer
Vision and Pattern Recognition Conference (CVPR), June 2025, pp.
7633–7642.
[43] K. Sayood, Introduction to data compression. Morgan Kaufmann, 2017.
[44] J. Conway and N. Sloane, “Fast quantizing and decoding and algorithms
for lattice quantizers and codes,” IEEE Transactions on Information
Theory, vol. 28, no. 2, pp. 227–232, 1982.
[45] A. Gersho and R. M. Gray, Vector quantization and signal compression.
Springer Science & Business Media, 2012, vol. 159.
[46] J. H. Conway and N. J. A. Sloane, Sphere packings, lattices and groups.
Springer Science & Business Media, 2013, vol. 290.
[47] E. Agustsson, F. Mentzer, M. Tschannen, L. Cavigelli, R. Timofte,
L. Benini, and L. V. Gool, “Soft-to-hard vector quantization for end-
to-end learning compressible representations,” Advances in neural in-
formation processing systems, vol. 30, 2017.
[48] Y. Bengio, N. L´eonard, and A. Courville, “Estimating or propagating
gradients through stochastic neurons for conditional computation,” arXiv
preprint arXiv:1308.3432, 2013.
[49] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation
learning,” Advances in neural information processing systems, vol. 30,
2017.
[50] Y. Takida, T. Shibuya, W. Liao, C.-H. Lai, J. Ohmura, T. Uesaka,
N. Murata, S. Takahashi, T. Kumakura, and Y. Mitsufuji, “Sq-vae:
Variational bayes on discrete representation with self-annealed stochastic
quantization,” arXiv preprint arXiv:2205.07547, 2022.
[51] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An
imperative style, high-performance deep learning library,” Advances in
neural information processing systems, vol. 32, 2019.
[52] L. Babai, “On lov´asz’lattice reduction and the nearest lattice point
problem,” Combinatorica, vol. 6, pp. 1–13, 1986.
[53] T. Chen and Z. Ma, “Variable bitrate image compression with quality
scaling factors,” in ICASSP 2020-2020 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2020,
pp. 2163–2167.
[54] Z. Cui, J. Wang, S. Gao, T. Guo, Y. Feng, and B. Bai, “Asymmetric
gained deep image compression with continuous rate adaptation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2021, pp. 10 532–10 541.
[55] K. Tong, Y. Wu, Y. Li, K. Zhang, L. Zhang, and X. Jin, “Qvrf: A
quantization-error-aware variable rate framework for learned image com-
pression,” in 2023 IEEE International Conference on Image Processing
(ICIP).
IEEE, 2023, pp. 1310–1314.
[56] F. Kamisli, F. Racap´e, and H. Choi, “Variable-rate learned im-
age compression with multi-objective optimization and quantization-
reconstruction offsets,” in 2024 Data Compression Conference (DCC).
IEEE, 2024, pp. 193–202.
[57] Y. Chen, M. Li, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Pcgs:
Progressive compression of 3d gaussian splatting,” arXiv preprint
arXiv:2503.08511, 2025.
[58] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[59] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
[60] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Bros-
tow, “Deep blending for free-viewpoint image-based rendering,” ACM
Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1–15, 2018.
[61] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[62] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
[63] G. Bjontegaard, “Calculation of average psnr differences between rd-
curves,” ITU SG16 Doc. VCEG-M33, 2001.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
[64] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d
scene representation via self-organizing gaussian grids,” in European
Conference on Computer Vision.
Springer, 2024, pp. 18–34.
[65] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, “Compgs:
Efficient 3d scene representation via compressed gaussian splatting,” in
Proceedings of the 32nd ACM International Conference on Multimedia,
2024, pp. 2936–2944.
[66] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, “Fast
feedforward 3d gaussian splatting compression,” in The Thirteenth
International Conference on Learning Representations, 2025. [Online].
Available: https://openreview.net/forum?id=DCandSZ2F1

<!-- page 14 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
14
–Supplementary Material–
TABLE V: Results of ‘HAC [14] + Ours SALVQ’ for each
scene from Deep Blending dataset [60].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
drjohnson
29.72
0.905
0.258
6.15
playroom
30.63
0.904
0.265
3.79
AVG
30.18
0.905
0.261
4.97
0.004
drjohnson
29.61
0.904
0.263
4.93
playroom
30.51
0.902
0.273
3.03
AVG
30.06
0.903
0.268
3.98
0.008
drjohnson
29.39
0.899
0.274
3.99
playroom
30.29
0.899
0.278
2.56
AVG
29.84
0.899
0.276
3.27
0.015
drjohnson
29.04
0.898
0.279
3.45
playroom
30.20
0.903
0.280
2.18
AVG
29.62
0.901
0.280
2.81
0.025
drjohnson
28.96
0.895
0.287
3.18
playroom
29.73
0.889
0.290
1.94
AVG
29.35
0.897
0.288
2.56
TABLE VI: Results of ‘HAC [14] + Ours SALVQ’ for each
scene from Tank & Temples dataset [59].
λr
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
train
22.69
0.820
0.210
8.10
truck
25.99
0.881
0.150
10.06
AVG
24.34
0.850
0.180
9.08
0.004
train
22.47
0.816
0.215
6.52
truck
25.92
0.878
0.157
8.02
AVG
24.20
0.847
0.186
7.27
0.008
train
22.42
0.809
0.227
4.82
truck
25.75
0.873
0.167
6.92
AVG
24.09
0.841
0.197
5.92
0.015
train
22.22
0.802
0.238
4.02
truck
25.58
0.871
0.176
5.66
AVG
23.90
0.837
0.207
4.84
0.025
train
22.06
0.792
0.253
3.19
truck
25.33
0.863
0.191
4.99
AVG
23.70
0.827
0.222
4.09
TABLE VII: Results of ‘HAC [14] + Ours SALVQ’ for each
scene from Mip-NeRF 360 dataset [58].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
bicycle
25.15
0.744
0.261
32.93
bonsai
32.83
0.946
0.183
9.74
counter
29.62
0.915
0.188
8.24
flowers
21.31
0.575
0.378
22.19
garden
27.42
0.847
0.143
26.16
kitchen
31.57
0.928
0.124
9.50
room
32.02
0.925
0.200
5.86
stump
26.65
0.763
0.265
20.57
treehill
23.36
0.646
0.351
23.97
AVG
27.77
0.810
0.233
17.68
0.004
bicycle
25.06
0.742
0.267
26.80
bonsai
32.40
0.943
0.188
7.94
counter
29.47
0.911
0.195
6.71
flowers
21.29
0.574
0.380
18.46
garden
27.31
0.842
0.153
21.12
kitchen
31.27
0.925
0.130
7.64
room
31.73
0.922
0.208
4.79
stump
26.60
0.760
0.273
16.66
treehill
23.29
0.644
0.358
18.62
AVG
27.60
0.807
0.239
14.30
0.008
bicycle
24.94
0.736
0.275
21.60
bonsai
31.85
0.937
0.197
6.63
counter
29.07
0.905
0.204
5.34
flowers
21.17
0.564
0.392
14.25
garden
27.05
0.832
0.167
16.35
kitchen
30.80
0.918
0.139
5.94
room
31.48
0.917
0.217
3.95
stump
26.51
0.756
0.283
13.66
treehill
23.19
0.638
0.370
14.71
AVG
27.34
0.800
0.249
11.38
0.015
bicycle
24.70
0.727
0.289
17.45
bonsai
31.26
0.932
0.205
5.85
counter
28.67
0.897
0.218
4.44
flowers
21.06
0.553
0.405
11.37
garden
26.65
0.817
0.191
13.70
kitchen
30.36
0.911
0.149
5.08
room
31.04
0.910
0.230
3.44
stump
26.31
0.745
0.302
10.97
treehill
23.15
0.627
0.388
11.97
AVG
27.02
0.791
0.264
9.36
0.025
bicycle
24.48
0.714
0.306
15.32
bonsai
30.66
0.925
0.215
5.39
counter
28.16
0.886
0.233
3.91
flowers
20.82
0.537
0.423
9.83
garden
26.42
0.805
0.207
11.66
kitchen
29.77
0.902
0.163
4.35
room
30.76
0.905
0.240
3.13
stump
25.98
0.725
0.328
9.36
treehill
23.04
0.614
0.406
10.43
AVG
26.68
0.779
0.280
8.15
TABLE VIII: Results of ‘ContextGS [18] + Ours SALVQ’ for
each scene from Deep Blending dataset [60].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
drjohnson
29.72
0.907
0.256
4.90
playroom
30.73
0.910
0.265
3.58
AVG
30.23
0.908
0.260
4.24
0.004
drjohnson
29.65
0.905
0.262
3.81
playroom
30.64
0.909
0.269
2.79
AVG
30.14
0.907
0.266
3.30
0.008
drjohnson
29.51
0.903
0.269
2.89
playroom
30.45
0.905
0.278
2.20
AVG
29.98
0.904
0.274
2.54
0.015
drjohnson
29.24
0.898
0.282
2.24
playroom
30.24
0.902
0.286
1.74
AVG
29.74
0.900
0.284
1.99
0.025
drjohnson
29.03
0.892
0.293
1.80
playroom
29.99
0.899
0.295
1.40
AVG
29.51
0.895
0.294
1.60

<!-- page 15 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
15
TABLE IX: Results of ‘ContextGS [18] + Ours SALVQ’ for
each scene from Tank & Temples dataset [59].
λr
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
train
22.74
0.823
0.211
7.58
truck
26.05
0.886
0.146
9.11
AVG
24.40
0.855
0.179
8.35
0.004
train
22.67
0.819
0.217
6.10
truck
26.02
0.885
0.150
7.53
AVG
24.35
0.852
0.184
6.81
0.008
train
22.48
0.812
0.228
4.88
truck
25.89
0.881
0.158
6.30
AVG
24.19
0.846
0.193
5.59
0.015
train
22.33
0.804
0.240
3.91
truck
25.76
0.878
0.165
4.84
AVG
24.05
0.841
0.202
4.37
0.025
train
22.07
0.793
0.253
3.11
truck
25.59
0.872
0.176
4.30
AVG
23.83
0.833
0.215
3.71
TABLE X: Results of ‘ContextGS [18] + Ours SALVQ’ for
each scene from Mip-NeRF 360 dataset [58].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
bicycle
25.03
0.739
0.265
25.24
bonsai
32.85
0.948
0.182
8.24
counter
29.57
0.915
0.191
7.42
flowers
21.27
0.574
0.378
19.50
garden
27.44
0.849
0.140
22.18
kitchen
31.45
0.928
0.127
8.34
room
31.85
0.925
0.202
5.51
stump
26.64
0.762
0.266
16.78
treehill
23.27
0.645
0.350
19.75
AVG
27.71
0.809
0.233
14.77
0.004
bicycle
25.05
0.738
0.270
20.64
bonsai
32.57
0.946
0.187
6.99
counter
29.34
0.911
0.198
5.97
flowers
21.28
0.575
0.377
16.09
garden
27.34
0.846
0.146
18.30
kitchen
31.27
0.925
0.133
6.70
room
31.71
0.923
0.208
4.24
stump
26.64
0.763
0.269
14.32
treehill
23.36
0.646
0.354
15.79
AVG
27.62
0.808
0.238
12.12
0.008
bicycle
25.01
0.736
0.277
17.42
bonsai
32.17
0.942
0.193
5.58
counter
29.13
0.907
0.205
4.68
flowers
21.24
0.572
0.383
12.66
garden
27.19
0.839
0.156
14.11
kitchen
30.88
0.920
0.140
5.20
room
31.50
0.919
0.216
3.43
stump
26.66
0.765
0.271
11.60
treehill
23.26
0.646
0.356
12.98
AVG
27.45
0.805
0.244
9.74
0.015
bicycle
25.02
0.732
0.286
13.66
bonsai
31.81
0.938
0.200
4.54
counter
28.75
0.901
0.215
3.76
flowers
21.27
0.569
0.389
10.55
garden
27.00
0.829
0.173
11.05
kitchen
30.52
0.914
0.149
4.17
room
31.20
0.914
0.227
2.74
stump
26.60
0.761
0.280
9.53
treehill
23.27
0.643
0.366
10.40
AVG
27.27
0.800
0.254
7.82
0.025
bicycle
24.82
0.722
0.301
11.42
bonsai
31.39
0.933
0.209
4.16
counter
28.50
0.894
0.225
3.17
flowers
21.07
0.563
0.397
8.77
garden
26.74
0.817
0.193
8.94
kitchen
30.12
0.907
0.162
3.58
room
30.96
0.907
0.241
2.32
stump
26.55
0.757
0.291
8.29
treehill
23.16
0.638
0.377
8.48
AVG
27.03
0.793
0.266
6.57
TABLE XI: Results of ‘HAC++ [17] + Ours SALVQ’ for each
scene from Deep Blending dataset [60].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
drjohnson
29.78
0.907
0.257
4.69
playroom
30.91
0.911
0.261
3.32
AVG
30.34
0.909
0.260
4.01
0.004
drjohnson
29.71
0.905
0.264
3.36
playroom
30.66
0.909
0.268
2.39
AVG
30.18
0.907
0.266
2.87
0.008
drjohnson
29.51
0.901
0.276
2.38
playroom
30.49
0.901
0.282
1.70
AVG
30.00
0.901
0.279
2.03
0.015
drjohnson
29.21
0.894
0.290
1.70
playroom
30.33
0.901
0.292
1.25
AVG
29.77
0.897
0.291
1.47
0.025
drjohnson
28.93
0.887
0.305
1.25
playroom
29.98
0.896
0.303
1.01
AVG
29.46
0.892
0.304
1.13
TABLE XII: Results of ‘HAC++ [17] + Ours SALVQ’ for
each scene from Tank & Temples dataset [59].
λr
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
train
22.61
0.820
0.213
5.95
truck
26.05
0.886
0.150
7.52
AVG
24.33
0.853
0.181
6.74
0.004
train
22.58
0.816
0.223
4.67
truck
25.94
0.882
0.157
5.77
AVG
24.26
0.849
0.190
5.22
0.008
train
22.33
0.804
0.240
3.49
truck
25.81
0.873
0.171
4.02
AVG
24.07
0.839
0.205
3.76
0.015
train
22.20
0.796
0.254
2.62
truck
25.59
0.869
0.187
2.83
AVG
23.89
0.832
0.221
2.73
0.025
train
22.09
0.784
0.274
1.98
truck
25.36
0.859
0.208
2.19
AVG
23.72
0.822
0.241
2.09

<!-- page 16 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
16
TABLE XIII: Results of ‘HAC++ [17] + Ours SALVQ’ for
each scene from Mip-NeRF 360 dataset [58].
λ
Scenes
PSNR↑
SSIM↑
LPIPS↓
SIZE↓
0.002
bicycle
25.16
0.738
0.276
17.50
bonsai
32.86
0.947
0.185
6.91
counter
29.64
0.915
0.193
6.24
flowers
21.30
0.573
0.384
14.38
garden
27.37
0.842
0.158
17.10
kitchen
31.53
0.927
0.130
7.09
room
31.99
0.924
0.207
4.56
stump
26.67
0.762
0.275
12.42
treehill
23.30
0.643
0.366
14.52
AVG
27.76
0.808
0.241
11.19
0.004
bicycle
25.07
0.731
0.291
12.74
bonsai
32.55
0.944
0.190
5.32
counter
29.42
0.910
0.202
4.75
flowers
21.29
0.569
0.392
10.76
garden
27.18
0.833
0.178
12.76
kitchen
31.27
0.923
0.137
5.25
room
31.77
0.920
0.217
3.45
stump
26.59
0.759
0.285
9.36
treehill
23.33
0.639
0.379
10.42
AVG
27.61
0.803
0.252
8.31
0.008
bicycle
24.97
0.721
0.306
9.20
bonsai
32.22
0.939
0.199
3.97
counter
29.14
0.903
0.213
3.52
flowers
21.24
0.565
0.401
7.99
garden
26.87
0.819
0.203
9.19
kitchen
30.81
0.916
0.147
3.81
room
31.57
0.915
0.229
2.58
stump
26.56
0.754
0.297
6.95
treehill
23.25
0.633
0.395
7.51
AVG
27.40
0.796
0.266
6.08
0.015
bicycle
24.80
0.707
0.326
6.40
bonsai
31.78
0.936
0.206
3.03
counter
28.83
0.895
0.229
2.61
flowers
21.16
0.556
0.414
5.77
garden
26.55
0.800
0.238
6.44
kitchen
30.33
0.908
0.162
2.77
room
31.25
0.908
0.244
1.88
stump
26.46
0.745
0.316
5.01
treehill
23.21
0.623
0.414
5.34
AVG
27.15
0.786
0.283
4.36
0.025
bicycle
24.65
0.690
0.347
4.74
bonsai
31.27
0.929
0.217
2.33
counter
28.42
0.882
0.250
1.97
flowers
21.04
0.544
0.430
4.34
garden
26.18
0.777
0.273
4.62
kitchen
29.83
0.898
0.180
2.01
room
30.94
0.900
0.262
1.47
stump
26.26
0.732
0.336
3.75
treehill
23.12
0.610
0.435
3.88
AVG
26.86
0.773
0.303
3.24
