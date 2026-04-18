<!-- page 1 -->
GSCache: Real-Time Radiance Caching for Volume Path Tracing
using 3D Gaussian Splatting
David Bauer
, Qi Wu
, Hamid Gadirov
, and Kwan-Liu Ma
Reference 
512 SPP
PT + Ours 
NEE 1 SPP
PT 
NEE 1 SPP
PT 
Uniform 1 SPP
41ms
12.4 PSNR
65ms
14.7 PSNR
38+14ms
16.9 PSNR
17s
-
Reference 
512 SPP
PT + Ours 
NEE 1 SPP
PT 
NEE 1 SPP
PT 
Uniform 1 SPP
25ms
12.0 PSNR
33ms
14.8 PSNR
23+9ms
17.3 PSNR
15s
-
Reference 
512 SPP
PT + Ours 
NEE 1 SPP
PT 
NEE 1 SPP
PT 
Uniform 1 SPP
29ms
10.9 PSNR
40ms
12.5 PSNR
23+3ms
15.9 PSNR
48s
-
Fig. 1: Our path-space cache improves image quality at low sample counts at comparable compute cost. We compare against a
path tracer (PT) using uniform sampling and a version implementing next-even-estimation (NEE). Cache rendering time is constant,
yielding increasing returns as sample counts increase. The method is non-invasive and easy to integrate into existing rendering
applications.
Abstract—Real-time path tracing is rapidly becoming the standard for rendering in entertainment and professional applications. In
scientific visualization, volume rendering plays a crucial role in helping researchers analyze and interpret complex 3D data. Recently,
photorealistic rendering techniques have gained popularity in scientific visualization, yet they face significant challenges. One of the
most prominent issues is slow rendering performance and high pixel variance caused by Monte Carlo integration. In this work, we
introduce a novel radiance caching approach for path-traced volume rendering. Our method leverages advances in volumetric scene
representation and adapts 3D Gaussian splatting to function as a multi-level, path-space radiance cache. This cache is designed to be
trainable on the fly, dynamically adapting to changes in scene parameters such as lighting configurations and transfer functions. By
incorporating our cache, we achieve less noisy, higher-quality images without increasing rendering costs. To evaluate our approach,
we compare it against a baseline path tracer that supports uniform sampling and next-event estimation and the state-of-the-art for
neural radiance caching. Through both quantitative and qualitative analyses, we demonstrate that our path-space radiance cache is a
robust solution that is easy to integrate and significantly enhances the rendering quality of volumetric visualization applications while
maintaining comparable computational efficiency.
Index Terms—Radiance caching, path tracing, volume rendering, gaussian splatting
1
Introduction
Photorealistic rendering of scientific volume datasets in real time
has become a feasible alternative to more traditional rendering
techniques in recent years. The advances in rendering research
and graphics hardware have enabled real-time path tracing
for many visual effects. These applications rely on low sample
counts, smart sampling strategies, and advanced post-processing
techniques to provide acceptable output quality.
Despite this, high-quality real-time rendering remains chal-
lenging in domains like volume rendering, which can be prone to
excess levels of Monte Carlo noise as a result of high sampling
variance. Noise mainly stems from variance in the sampling
distribution and the need to sample particle interactions with
the volume—a process not required in surface-only path trac-
• David Bauer and Kwan-Liu Ma are with the University of
California at Davis. E-mails: {davbauer, klma}@ucdavis.edu
• Qi Wu is with NVIDIA Research E-mail: qiwu@nvidia.com
• Hamid Gadirov is with the University of Groningen E-mail:
h.gadirov@rug.nl
ers. In extension, variance is influenced by the quality of the
samples we take. Past research has focused on different ways
to address these issues by improving the sampling scheme to
create more relevant samples, making the few samples we take
count more [4,28,29,38]. Orthogonal approaches like radiance
caching can be used to store previously seen radiance samples
in a cache data structure, which can then be used to augment
subsequent samples [10,23,34,47]. Many times, these methods
are highly specialized solutions that require much additional
work to be integrated into existing renderers. Despite these ad-
vances, scientific volume rendering has received little attention
in these domains, making it an interesting subject for further
improvement.
In this work, we introduce a real-time path-space cache for
volume rendering that stores attenuated radiance at different
path lengths and can quickly adapt to changes in the scene,
such as lighting, transfer function, or slicing operations. The
cache is represented as a multi-level collection of 3D Gaussians.
Each level caches the attenuated path-space radiance for paths
of a specific length. Cache sizes are sub-sampled from higher
to lower levels, similar to the construction of MIP levels in
texture processing. This is done since paths of length n generally
contribute more to the final image than paths of length m, where
arXiv:2507.19718v2  [cs.GR]  2 Aug 2025

<!-- page 2 -->
m > n. This property allows us to allocate more computational
resources and representational power to paths that have a higher
image contribution. The cache is optimized in real time with
noisy samples from the renderer. The only information needed
from the rendering application is the attenuated radiance value
and the path length that produced it.
We evaluate our approach by comparing it to a baseline vol-
ume path tracer. In addition to image quality and rendering
speed analysis, we conduct an ablation study to justify our
design. We test our approach on a scientific volume renderer
that implements path tracing with geometry-based lighting and
compare results from uniform sampling and using next-event
estimation (NEE). Our results show that Gaussian radiance
caches are an effective, fast, and easy-to-integrate alternative
to world- and image-space-based radiance caches. Our contri-
butions can be summarized as follows.
• We introduce a novel radiance cache optimized for volume
rendering that caches path-space radiance using multiple
levels of Gaussian splats.
• The cache works in real time on complex datasets and in
a wide variety of use cases and adapts quickly to changes
in the transfer function and lighting parameters, which
improves the overall image quality and rendering times.
• Optimizing the cache is possible not only with clean sam-
ples but also with noisy data, as is commonly found in
Monte-Carlo-based renderers.
• The path-space nature of the cache and its non-invasive
design make it easy to use and integrate into existing
rendering solutions.
With these contributions, we hope to support the design and
quality of scientific renderers and encourage the use of radiance
caching in scientific applications and beyond.
2
Related Work
This work is related to methods in scientific visualization, vol-
ume rendering, radiance caching, and machine learning. In
the following paragraphs, we discuss relevant works from these
fields.
2.1
Radiance Caching
Radiance caching aims to trade variance for bias to improve the
image quality of path tracing applications. In the context of
path tracing, variance is the result of sampling decisions along
a path, while bias is generally the result of making assumptions
about radiance distributions in a scene that are not analytically
tractable.
In most cases (ir-)radiance is cached by storing
radiance probes in world-accessible spatial data structures that
can be sampled during rendering [13, 23, 44]. Recently, some
approaches have made use of deep learning to store and access
radiance information [2,10,34]. Early approaches usually focus
on caching indirect illumination after the first bounce. This
means that all paths that terminate into the cache do so after a
single bounce. A downside of such approaches is that using the
cache so soon can introduce considerable error from bias. This
is why other approaches try to mask the bias by allowing longer
paths [17,34]. There has also been work specifically targeting
volume rendering. Khlebnikov et al. [19] introduce a parallel
GPU-based irradiance caching approach for interactive volume
rendering. Jakob et al. [12] use hierarchical Gaussian mixture
models to derive scene radiance from photon traces and Šmajdek
et al. [46] introduce radiance caching to address mixed-modality
scenarios where isosurface and volume rendering are combined.
One crucial consideration to make when allowing for differ-
ent path lengths is to decide when to terminate a path and
instead to read stored radiance from the cache and update its
entries. Throughout the literature, several heuristics have been
developed to address this issue. For example, Müller et al. [34]
terminate paths based on their spatial spread using a heuristic
proposed by Baekert et al. [3]. Kandlbinder et al. [16] develop
a variance-based heuristic that offers better control of the bias
error bound when sampling the cache.
2.2
Scientific Volume Rendering
Volume rendering is vital to the field of scientific visualization.
Broadly speaking, applications of scientific rendering can be
categorized into two groups—visualization for analysis and visu-
alization for illustration. The former group relies on rendering
techniques that offer interactive framerates and user interac-
tions, such as cutting planes and transfer function editors. We
commonly see direct volume rendering (DVR) as the rendering
method of choice in this group. Illustrative visualization, on the
other hand, prioritizes image quality over interactivity and cus-
tomization. It is closer, in purpose, to more traditional offline
rendering scenarios. In this case, path tracing with physically-
based illumination models is preferable. Both types of rendering
and the underlying optical models for volume visualization have
been aptly summarized by Max et al. [32]. Despite promises
of realism, calculating physically-based volumetric illumination
via the radiative transfer equation (RTE) is expensive. For
a long time, research has focused on finding appropriate ap-
proximations to enable real-time rendering of scientific datasets.
Jönsson et al. [15] provide an excellent summary of this category
of techniques. In the following, we highlight some notable works
in this area. Gradient-based methods were one of the first ways
to render volume datasets. Levoy et al. [27] use volume gradients
to apply surface shading models like the Blinn-Phong reflection
model. Local shading is a simple step up from gradient-based
rendering. Ambient occlusion techniques [41–43] have been
popular in the domain as they are relatively easy to implement
and grant viewers a greater sense of depth, which makes it easier
to discern details in complex or abstract volume datasets. Next,
slice-based methods like Kniss et al.’s half-angle slicing [21,22]
provide a more sophisticated model and were one of the first
to simulate multi-scatter illumination. Lastly, physically-based
particle tracking [51] and path tracing allow for high levels of
realism with full global illumination. Here, Zhang et al. [56]
use pre-computed photon maps while others [6,24] work on ray
tracing-based approaches. Jönsson et al. [14] expand on the
idea of photon mapping for scientific visualization and introduce
an approach that selectively recomputes photon traces based
on transfer function changes. Lastly, Bauer et al. [2] transform
photon maps into implicit neural representations, which im-
proves rendering speeds, enabling interactive visualization. Our
work is related to these techniques in that we aim to provide
higher performance and lower variance for scientific path tracing
applications.
2.3
Machine Learning for Scientific Visualization
Scientific visualization research has been comparatively slow to
adapt machine learning and deep learning into its repertoire.
Nevertheless, recent years have witnessed several interesting
applications of machine learning techniques to address pervasive
problems in scientific visualization and rendering.
Finding good transfer functions and viewing parameters has
been a longstanding challenge in scientific visualization. Weiss
et al. [50] introduce a differentiable direct volume rendering
algorithm that addresses this issue. Using their method, param-
eters like the input transfer function, volume density, or camera
configuration can be optimized via gradient descent. Later, Pan
et al. [39] expand on this idea and introduce an interactive tool
to explore the latent space of different transfer function designs.
Users can modify designs and optimize for the corresponding
transfer function using Weiss et al.’s [50] approach.
Real-time rendering is particularly challenging in scientific
visualization as datasets tend to be large and complex. Develop-
ing rendering applications for such data requires careful design

<!-- page 3 -->
to operate within the memory and performance limits. This
becomes especially pressing when advanced illumination models
like ambient occlusion (AO) or multi-scatter path tracing are
used.
Engel et al. [7] introduce a neural representation for
real-time volumetric ambient occlusion rendering, improving
baseline costs of calculating AO. Bauer et al. [2] extend this
thought to full global illumination. Their method introduces
a phase-function varying neural representation of volumetric
lighting that can be used to replace traditional path sampling
beyond the first interaction, reducing noise and improving ren-
dering speeds. Aside from caching approaches, there have been
efforts [1,49] to reduce rendering costs by adaptively sampling
and reconstructing data in the volume rendering pipeline.
Lastly, storage poses additional challenges over traditional
rendering applications.
Volume datasets tend to be large,
with multiple parameter grids and even multiple volumes com-
posed as a time series. Neural networks have proved to be a
powerful tool to implicitly represent and compress such vol-
umes [11,48,53]. Weiss et al. [48] and Wu et al. [53] develop
implicit neural representations (INR) for volume compression
to address this issue. Zavorotny et al. [55] address INR ren-
dering bottlenecks while, more recently, Wu et al. [52] and
Gadirov [9] have extended neural visualization methods using
hypernetworks for parameter-space exploration.
3
Method
Our method broadly consists of two parts. The first is a path-
space radiance representation attained by training our Gaussian
cache. The second component is the integration of our cache
into a volume path tracer for real-time rendering and cache
optimization. This section describes our cache design in detail.
3.1
A Path Space Radiance Cache
Traditionally, the volumetric radiance transfer can be charac-
terized by the integral of transmittance-attenuated radiance
accumulated along a sampling ray (see Equation 1).
L(x,ω) =
Z ∞
0
Tr(x′ →x)Ls(x′,−ω)dt
(1)
where Tr is the transmittance between two points x and x′ and
Ls is the source term describing the amount of radiance going
through point x in direction −ω. This radiance integral can be
reformulated as an integral in path space. In this formulation of
radiance, we consider sets of possible paths of length n. The final
color of a pixel can be found by adding up the contribution of all
path sets n ∈[0,∞] (see Figure 2). For a detailed examination
of the path space radiance formulation, please refer to Chapter
8 in Veach [45]. Our cache design is based on this same notion
of path space radiance. We represent the total radiance of all
paths of length n in a single cache level. For each i ≤n we define
a separate cache level to hold the radiance of that subspace of
path space radiance (see Figure 3).
Each cache level is represented as a point cloud of three-
dimensional Gaussians. These Gaussians can be used to recon-
struct the path space subset radiance for each level by rasterizing
them into images using the recently introduced Gaussian splat-
ting rasterization technique [18]. One notable feature of 3D
Gaussian splatting is its fast differentiable rasterizer, which
allows for splatting at high frame rates. We make use of this
property to enable fast access to radiance at each cache level.
Before every path tracing pass, a new set of cache images is
rasterized and passed to the path tracing algorithm. After one
pass in the path tracer, the cache is optimized with fresh path
samples. In the following sections, we describe how the cache is
initialized, how paths are terminated into the cache, and how
it is optimized in real time during rendering.
3.2
Cache Initialization
We initialize the radiance cache by doing a single sampling pass
over the volume data. The goal of this step is to generate a
point cloud that will be used to initialize each cache level with
a set of Gaussians that roughly correspond to the structure
of the dataset. To this end, we randomly generate rays that
intersect the volume. The rays are traced through the dataset
using delta tracking [51]. If there is an interaction, we record
the position and albedo of the interaction and terminate the
process. In cases where the tracking algorithm does not produce
a valid interaction (i.e., when the ray leaves the volume), we
repeat the process with a newly generated random ray until
we find a valid point sample. For our purposes, we generate
an initial set of N points to initialize the cache. In our tests,
we found N = 300k to be a suitable initialization size. Upon
initialization, the point cloud is replicated and logarithmically
sub-sampled for each level, allotting more points to earlier levels
in the cache. This process results in a total number of PK
i=0
N
2i
points where K +1 is the number of levels chosen for the cache.
Based on the point locations and colors, we initialize a set
of Gaussians similar to Kerbl et al. [18]. We use a k-nearest
neighbor query to determine the initial scales of the Gaussians.
Unlike Kerbl et al. [18], we use a smaller initial scale for the
Gaussian covariances. Additionally, we cap any outliers with
a z-score greater than 2 to at most two standard deviations
above the mean to exclude unreasonably large Gaussians in our
starting condition.
si =
(µN +2∗σN)∧1
3
P2
j=0 dij
2
(2)
where si is the i-th Gaussian’s isotropic scale, dij is the distance
to j-th closest neighbor of the i-th point, and µN and σN
are the mean and standard deviation over all N points’ 3-
closest neighbor distances. Effectively, this process eliminates
large outliers and scales the overall set of Gaussians to 50%.
Leaving enough space between Gaussians helped stabilize cache
optimization on noisy input data.
3.3
Path Termination Heuristics
For caching to be efficient, it is common to introduce a path
termination heuristic to determine if a path will contribute a
meaningful amount of radiance. Several heuristics have been
used in surface path tracing for radiance caching [3, 16]. No-
tably, Kandlbinder et al. [16] introduced effective heuristics
to terminate paths that are unlikely to contribute significant
radiance to the integral. Müller et al. [34] used methods by
Bekaert et al. [3] for neural radiance caching. Unlike much
of prior work [2, 10, 13, 23, 34], our method operates in path-
space. In addition to that, our cache is specifically targeted
at volume rendering applications—an area that has received
little attention in prior work. These facts change the functional
requirements of the heuristic.
The overall goal of such a path termination heuristic is to
determine when the benefits of tracing a full, unbiased path
outweigh the benefits of using a biased cache sample instead.
Traditionally, an effective measure to determine this has been
to estimate the current path’s contribution if it were to leave
the volume after the current path vertex to receive radiance
from the environment outside the volume. In the case of basic
uniform path tracing, this estimation occurs when the path nat-
urally terminates, which happens when the tracking algorithm
does not find a valid interaction with the volume, indicating
that the path has left the volume bounds. This approach is
easy to implement but suffers from the fact that paths have
to be traced to their natural conclusion, foregoing the poten-
tial savings of terminating a low-contribution ray early. To
improve upon the naïve approach, we leverage the commonly
implemented technique of next-event-estimation (NEE). This

<!-- page 4 -->
+
+
=
All L1 Paths
All L2 Paths
All L3 Paths
v0
v1
v1
v2
v0
v0
Volume Dataset
All Possible Path Sets 
L ∈ [1, ∞]
Final Image
L1 Path
L2 Path
L3 Path
Light Source
Viewer
Path Space Representation
Volume Path Tracing
Image
Plane
Fig. 2: The path tracing integral can be characterized as the collection of all possible paths of all possible lengths. When combined, these paths
form the final image. Left: Example of a volume path tracer. Primary rays are produced by the viewer and interact with the volume at path vertices
vi. The number of interactions with the volume before a path terminates determines the path length. Right: Paths of the same length can be
grouped into path-space sets.
technique estimates direct light contribution at every path ver-
tex. The light sample is then combined with the continued path
contribution using multiple importance sampling. Having direct
light samples at every path vertex allows us to continuously
estimate a path’s contribution as we trace it through the vol-
ume. In our implementation, we distinguish two cases—natural
and early path termination—and we propose two simple cache
sampling strategies for both cases.
3.3.1
Natural Path Termination
The base case in most volume path tracers is reached when
there is no new valid interaction with the volume, and the
path naturally leaves the volume. Since path termination is
not deliberately sampled but rather an unforeseeable outcome
of phase function sampling, terminating directions are chosen
according to the volume’s scattering behavior. The majority of
such terminal path segments typically does not intersect with
any of the light sources in the scene, leading to zero radiance
contribution along the ray, effectively wasting the whole path
sample. In scenes with infinite area light sources, samples might
contribute a small amount of radiance. We can make use of
this fact to sample our cache. If the path terminates with a
non-zero radiance contribution, we use the unbiased sample;
otherwise, we determine the cached radiance at the terminating
path length and use this value instead. In this way, every path
that terminates contributes a meaningful amount of radiance
to the final image.
3.3.2
Early Path Termination
In addition to sampling the cache for naturally terminated
paths, we also consider terminating paths early into the cache.
This has the benefit that we not only get a less noisy sample
but also do so at a performance gain, as reading the cache is
cheaper than sampling the path to its natural termination.
The heuristic uses the luminance of the current path through-
put Tr to generate a termination probability p (see Algorithm 1).
The Algorithm takes a list of path vertex albedos σi to com-
pute Tr. At the same time, we keep track of βi to importance
sample cache hits. The details of the algorithm are covered
in the subsequent paragraphs. The heuristic is similar to the
Russian Roulette termination technique that is commonly used
in path tracers to cut long paths with little contribution short.
Unlike Russian Roulette, our method uses a much higher ini-
tial throughput threshold to become active. In our tests, we
consider any throughput values less than 0.9. The reason for
having this threshold is to guarantee that a fraction of paths
will continue regardless of any cache sampling probability to
ensure that we can gather paths of all lengths. Additionally,
we employ a user-defined termination coefficient C, which mod-
ifies the baseline probability in favor or against the current
throughput. The final probability is used to determine if the
current path should be terminated into the cache and, if not, is
used to importance sample the continuation of the path (see
Section 3.4).
Algorithm 1 Early path termination heuristic
Input [σ1,...,σn], C, βn
Output Trout,βn+1
Trout ←Qn
k=1 σk
Tr ←clamp(C ·luminance(Trout),0,1)
if Tr < 0.9 then
q ∼U(0,1)
p ←1−Tr
if q < p then
return True
end if
Trout ←T rout
T r+ϵ
βn+1 ←βnTr
end if
return False
3.4
Cache Sampling
L1
L2
L3
Cache
v2
v1
v0
Viewer
Fig. 3: Illustration of our cache sampling and path termination mech-
anisms. Using our path termination heuristic allows us to sample the
cache at different levels depending on the throughput as the product of
all prior path vertex albedos σi and the cache sampling coefficient C.
If a path is terminated at depth n, we want to determine
if the final radiance should come from the cache or the path’s
contribution until vertex n. To make this decision, we look to the
current total path throughput Tr as an indicator of anticipated
path contribution (Figure 3). We sample the radiance cache
based on the anticipated path contribution to select either the
value in the cache at the corresponding level or use the current
unbiased radiance that we accumulated through next-event
estimation.

<!-- page 5 -->
3.4.1
Throughput Importance Sampling
The considerations we make for natural path termination are
sufficient to sample the cache correctly. However, when paths
are terminated into the cache early, we need to make adjust-
ments to keep the overall path sampling unbiased. First, we
adjust the throughput Trout for cases where the cache sampling
resulted in no hit. This is equivalent to adjusting the path
throughput in Russian roulette. The effective path throughput
Trout is importance sampled by dividing by the cache sampling
probability Tr (see Algorithm 1). Aside from this, we need to
consider what happens when there is a cache hit. In the follow-
ing, we describe how we ensure that cache reads contribute an
appropriate amount of radiance to the sample.
3.4.2
Cascaded Cache Radiance Importance Sampling
Fig. 4: Accounting for the cascaded cache sampling probability is neces-
sary to accurately attenuate cached radiance samples. (Left) Reference
image. (Top) rMSE error map of the rendering disregarding cascaded
cache sampling probability β. (Bottom) rMSE error map of the rendering
with correct β attenuation. Correcting for β clearly improves the image
error in the lower image compared to the converged reference.
Adjusting path throughput Trout by the probability of a
cache hit is not enough to ensure correct cache sampling. An-
other issue arises when there is a cache hit. The cached values
are pre-attenuated and directly represent the radiance at path
length n. However, this value does not take into account the
possibilities of early terminations at length i ∈[1,n −1] (Fig-
ure 4). Since paths could have terminated at any path vertex
before the current one, only a fraction of paths reach length n.
We need to account for this fact by keeping track of the product
of all prior cache sampling probabilities Tr and adjusting the
cached radiance by this product (see Algorithm 1). Not doing
so will result in images that are generally darker than the un-
biased reference. To keep track of the product path sampling
probability β, we initialize it to 1 and multiply it by Tr any
time the cache is missed. When the cache is hit at depth n, we
divide the cached radiance by βn−1 to account for the fact that
the cache was missed n −1 times. The effective attenuation
applied to a path sample is as follows.
ˆLn = Ln
Qn
k=1 σk
βn−1
(3)
Where the product is the regular path attenuation accumu-
lated from sampled albedos and βn−1 is the product of sampling
probabilities Tr up until depth n−1 (see Algorithm 1).
3.5
Cache Training
The radiance cache is trained in real time, and training can be
toggled as needed. To train the cache, we collect unbiased sam-
ples from the renderer, attenuate them, and assign them to their
appropriate cache level (Figure 5). In the case of both natural
and early path termination, the sample is simply attenuated
by Trout (see Algorithm 1) and stored in an intermediate path
buffer. We maintain K intermediate path buffers—one for each
level in the cache. At the time of termination, we determine the
current path length, which determines the intermediate buffer
that the sample gets assigned to.
+
+
Loss Gradient
Path Space Radiance
Gaussian Radiance Cache
Gaussian Splatting
Optimization
Cache Sampling
Fig. 5: Our cache is trained in real time on paths obtained during the
training cycle. The optimizer uses Gaussian splatting and noisy path
samples to optimize individual cache levels. At the same time, cached
path radiance is sampled during rendering to improve image quality and
runtime.
At the end of one sample pass, each pixel will have at most
one valid entry across the intermediate path buffers. At this
time, the samples are attenuated, meaning that their total
path throughput is taken care of. However, these samples are
ultimately noisy as they represent the radiance from only one
possible path of length n through the volume. Conventionally,
Gaussian splatting applications use clean target images, such
as photographs, to calculate gradients for optimization. This
poses a potential problem for our application.
We make the important observation that the expected value
of samples in the buffers behaves analogously to regular path
tracing, where paths of all lengths are collected indiscriminately.
This means that if we were to take an infinite number of samples,
we could accumulate a clean image for each of the buffers
representing the radiance contribution in path subspace n.
With this observation, we apply the principle of learning clean
data from noisy targets, which was first outlined by Lehtinen
et al. [26]. This work shows that gradient descent optimization
can generally be applied to problems with noisy targets if the
corrupted target’s expected value is the noise-free limit. Since
this is the case for our path space buffers, we can apply this
theory to enable training on noisy input buffers. This allows
us to use the intermediate buffers as target images to compute
the loss for the gradient calculation during the inverse splatting
step of the Gaussian rasterization algorithm.
3.6
Optimization and Hyperparameters
The original implementation of 3D Gaussian splatting [18] in-
cludes various hyperparameters to steer the optimization to
scenes with sparse input views. In our case, a lack of data is not
the problem. Therefore, we can fine-tune the original algorithm
to work more efficiently for our use case. In the following, we
describe various additions and changes that we made to the
optimization algorithm.
Spherical Harmonics. The original 3D Gaussian splatting
implementation [18] proposes the use of spherical harmonics
(SH) to capture view-dependent effects such as specular reflec-
tions. While a reasonable addition for traditional computer
vision tasks, Mallick et al. [31] show that a considerable amount
of time is spent on the optimization of SH coefficients. Since
view-dependent effects are not a driving factor in scientific vol-
ume visualization, we set the SH degree to 0, which effectively

<!-- page 6 -->
reduces the optimization problem to three view-independent
dimensions (RGB).
Densification and Pruning. We do not use the original
densification and pruning strategies introduced by Kerbl et
al. [18]. The noise of our training samples makes the splitting,
cloning, and pruning mechanisms too unstable for real-time
training. Furthermore, removing these heuristic-based strategies
improved not only stability but also the runtime of our method.
Loss. The data we are fitting has a high dynamic range
(HDR) with potentially unbounded radiance values. This is in
contrast to most scene representation tasks, which will draw
samples from LDR sources like photographs. Commonly used
loss terms like the L1 loss are ill-suited to adapt to HDR data as
very bright samples generate disproportionately large gradients,
which can overshadow the contribution of other samples. To
address this, we adapt the HDR loss proposed by Lehtinen
et al. [26], which contains a normalization term and reads as
follows.
Lhdr =
(ˆx−ˆy)2
k(ˆy +0.01)2
(4)
where ˆx is the expected target value (i.e., the noisy path sample)
and ˆy is the predicted sample generated from the Gaussian
splatting process. As inputs are images, we average the loss
over all pixels k.
Adaptive Learning Rates. Each parameter type is as-
signed a separate learning rate. Please refer to our evaluation
for specific starting values used in our trials. Since we gener-
ate a well-fit point cloud to initialize our cache levels, we rely
less heavily on the algorithm’s capability to optimize mean,
covariance, and scale. Furthermore, we employ a learning rate
schedule that is reflective of the user’s interaction with the
rendering system. When there is no change in the viewport,
we gradually reduce the learning rate to locally improve the
quality of the cache. In practice, we scale the learning rate by
the logarithm of the number of frames that the current viewport
has been observed (Equation 5).
ηt = η0
1
1+log(t)
(5)
If the viewport is changed, the learning rate is reset to its
initial level, allowing for quick adaptation to changed inputs.
The longer the viewer resides, the smaller the gradient-based
updates become and the better the fit becomes locally, which
improves the cache quality for the current viewport and its
immediate neighbors. Although all learning rates are scaled
in the same way, an adaptive algorithm that respects each
parameter’s gradients might be an interesting future addition.
Regularization.
In contrast to most other scene repre-
sentation approaches, our cache stores radiance values with
unpredictably high variance. This impacts the performance
of the optimization steps. Gradients can take on large values,
which can occur very sparsely in high-variance scenarios. All
this leads to instability in the training. In our tests, this specif-
ically manifested as flickering and exploding Gaussians whose
size quickly exceeded the scene scale and whose color dominated
the image. To address this issue, we introduce regularization to
penalize the emergence of large parameter values. Specifically,
we use the AdamW [30] optimizer, which regularizes parameters
independent of their adaptive step sizes. This modification en-
sures stability during training and prevents the aforementioned
artifacts.
3.7
Implementation
We implement the radiance cache using C++ and CUDA. Op-
timization is handled via the PyTorch C++ API [8] and is
directly integrated with the C++ runtime.
We expose the
caching library via a C-style API, which allows it to be easily
adapted to various applications and languages.
For our evaluation, we integrate the cache into a scientific vol-
ume path tracer written in C++ using NVIDIA OptiX 7.3 [36]
as a raytracing backend. Rendering kernels and caching inte-
gration are written in CUDA.
4
Evaluation
We evaluate our method in terms of runtime and image quality.
Our results are compared against renderings using a baseline
path tracer that implements uniform sampling and next-event
estimation (NEE), all of which use an isotropic phase function.
We also compare against state-of-the-art neural radiance caching
(NRC) [34]. Furthermore, we provide additional data and an
ablation study in the supplemental material.
4.1
System Specifications and Setup
All tests were run on the same machine under equal conditions.
The workstation used for these tests features an Intel Core
i7-6900K CPU with 64GB of memory and an NVIDIA TITAN
RTX with 24GB of video memory. The system was running
Ubuntu 22.04.3 LTS, and all components were compiled using
GCC 11.4.0 and NVCC 12.6. In our evaluations, we use a com-
mon set of settings and parameters for our renderer and caching
algorithm unless otherwise specified. Image resolution for per-
formance evaluation was chosen at 720p and at 1280×1024 for
quality comparison to better frame datasets. Rendering was
done at 1 sample per pixel (SPP) unless otherwise indicated.
We initialize the cache using N = 300k samples with subse-
quent cache levels receiving an exponentially sub-sampled set
of the initial points (see Section 3.2). In our setup, using three
cache levels, this results in a total cache size of 29400000 bytes
(approx. 28MB) for each scene and an average initialization
time of 411.15ms. See the supplemental material for a detailed
analysis.
For our tests, we use the following initial values for the
learning rates (LR). Point position LR is 1.16e−3, color LR is
1.25e−2, rotation LR is 1e−3, scaling LR is 0, and opacity LR
is 1.5e−1. We implement a version of NRC [34] for comparison
to our method. To this end, we gather NEE samples as training
data, use the same path termination heuristic as our method,
and disable self-training for a fair comparison. Since volume
radiance beyond the first bounce is highly diffuse, we use sample
positions as input, as using spatio-directional inputs resulted
in lower image quality in our tests. The model configuration is
the same as NRC [34]. However, we use the hash-grid encoding
for better positional encoding. The total size of the NRC cache
in our setup is 284999680 bytes (approx. 272MB).
4.2
Datasets
Our method is evaluated on a variety of volume datasets. In the
following, we describe each dataset’s characteristics (see Table 1).
We chose the datasets for this study to show a representative
selection of data from medicine, biology, physics, and other
fields. Since the present study concerns real-time applications,
we did not consider extreme-scale datasets, which would require
specialized data streaming and rendering techniques and might
limit the system to offline rendering, weakening the justification
for a cache.
For each of the datasets, we use a predefined
1D transfer function which remains unchanged throughout the
evaluation.
4.3
Visual Quality
We evaluate the visual quality of results obtained from our
method by comparing them to uncached results. The baselines
comprise results from a volume path tracer using uniform sam-
pling and another version using next-event estimation. Each
scene includes a single volume dataset with a custom transfer
function (see Table 1). The scenes are lit by a two-point lighting

<!-- page 7 -->
Fig. 6: Visual quality of our method compared to the baseline path tracer. We show results for images at 1 SPP and compare our method
(GSCache) against a baseline volume path tracer with uniform sampling (Uniform), a version that uses next-event estimation (NEE), and our
implementation of NRC [34].
Table 1: We use volume datasets from medical scans, CT scans, and
scientific simulations to test our method.
Data ranges in size and
complexity.
Dataset
Dimensions
Data Type
FullBody
512×512×1299
uint8
MechanicalHand
640×220×229
float32
Supernova [37]
432×432×432
float32
Carp
128×128×256
uint8
Zebrafish
592×413×956
float32
Spider
957×1195×1003
uint16
setup using two spherical area light sources placed outside the
volume. We choose small area lights to highlight the utility
of radiance caching as scenes with ambient or environment
lighting suffer less from sampling noise due to broader radiance
coverage.
In Figure 6, we compare the image quality of renderings
from several different scenes. The results show that our method
produces significantly higher image quality at low sample rates.
Images are less noisy, as evidenced by the generally higher PSNR,
and overall image quality notably exceeds that of baseline ren-
derings and state-of-the-art caching [34] results at equivalent
sampling rates. We highlight the cache quality over time in
Figure 7. After a cold start, it takes less than 16 samples for
the cache to adapt to a point where it outperforms the baseline
renderer in terms of visual quality. As rendering progresses, we
significantly improve over the baseline and achieve superior im-
age quality. Lastly, we investigate the effects of different cache
sampling coefficients C, which we define in Section 3.3.2. The
results show how C can be used to move along the bias-variance
scale to find a suitable balance between image quality, render-
ing speed, and unbiasedness (Figure 8). In practice, the lower
bound C = 0.0 is likely not going to be used as it corresponds
to pure cache sampling without introducing unbiased paths,
which implicitly precludes the possibility of cache training. We
found a value of C = 0.5 to be an adequate setting in our test
scenes. Furthermore, we discuss the implications of different C
values for rendering performance in the next section.
4.4
Runtime Performance
Runtime performance is an important factor for the use of a
radiance cache, as we generally want to achieve higher image
quality in the same time or less compared to the uncached
application.
We provide itemized runtime numbers in Table 2. Results
show that timings are overall comparable to NRC [34] and that
while there is a runtime overhead to rendering the cache every
frame, we generally make up for that time by cutting longer
paths short and terminating them into the cache. Naturally,
the sampling constant C influences the exact performance gain
from shorter paths (see Figure 9). We found a value of C = 0.5
to be a good balance between image quality, bias, and runtime
performance (see Figure 8). Note that the number of lights and
choice of transfer function can impact rendering performance
overall but does not have any meaningful impact on caching
times as we use the same number of cache points regardless
of the lighting environment and do not employ splitting. This

<!-- page 8 -->
Table 2: Frame timings of our method on different datasets captured from a screen-filling camera fly-through of 200 frames. We allowed for a
40-frame warm-up period to initialize the cache and show path tracing time (PT) for all four methods and splatting time (ST) , inference time (IT),
and optimization time (OT) for our method and NRC [34]. IT includes a composition pass to back-propagate and compose cached radiance. The
notation PTnee∗denotes NEE with the addition of our path termination heuristic and PTnee∗∗additionally adds path records and training sample
collection for NRC [34]. All values were captured at C = 0.5 with N = 300k initial points. All timings are in milliseconds.
Dataset
Path Tracing
Path Tracing + Ours
Path Tracing + NRC
PTuniform
PTnee
PTnee∗
ST
OT
PTnee∗∗
IT
OT
FullBody
28.92
40.39
22.72
3.12
15.78
32.00
4.29
29.74
MechanicalHand
24.62
33.29
23.36
8.94
33.48
32.51
3.95
26.85
Supernova
23.21
42.80
28.85
10.29
13.38
35.10
4.25
28.70
Carp
41.17
65.47
37.86
14.31
23.78
53.74
3.83
20.95
Zebrafish
64.83
151.43
85.19
7.88
24.61
120.19
3.12
22.10
Spider
71.37
148.01
81.86
24.00
77.89
119.61
4.18
26.31
Overall
42.35
80.23
46.64
11.42
31.49
65.53
3.94
25.78
Fig. 7: Comparison of cache state after a cold-start training for 64 frames.
Frame 1 shows the freshly initialized state with raw albedo colors. Over
the next few frames, the cache quickly adapts to the scene. We show
the cache states after 1, 4, 16, and 64 frames and juxtapose them
with an uncached frame capture. Images are rendered at 16 SPP, and
quality metrics are computed compared to the unbiased reference image.
Note that at 16 SPP, the cache provides diminishing returns as more
unbiased paths are accumulated. Nevertheless, our cache outperforms
the unbiased renderer at the same number of samples.
means the user-defined cache size stays constant throughout
the application life cycle. Training times can make up for a
significant portion of the splatting-related workload, which can
be alleviated by selectively enabling training when there are
significant changes in the scene or the viewport. For example,
training could be disabled if the learning rate crosses a lower
limit threshold due to a user residing in the same viewport for
prolonged periods of time or the optimization rate could be
adjusted (i.e., varying the number of frames after which the
optimization routine is invoked) based on the differential change
in camera position and viewing angle to avoid optimizing for
viewports that might be outdated in just a few frames. We
note that our implementation is not highly optimized, leav-
ing potential performance improvements for the optimization
step by leveraging custom low-level implementations. Please
Fig. 8: Comparison of using different values for the user-defined sampling
coefficient C at 1 SPP. Lower values favor termination into the cache,
while higher values lead to longer paths. The figure shows the trade-off
between bias and variance that can be made by choosing different values
for C. While lower values for C generally increase image metrics, they
also introduce more bias into the final image.
Fig. 9: Influence of the cache sampling probability C on rendering times.
Data was recorded on the FullBody dataset at 1 SPP.
refer to our discussion for potential performance improvements
suggested by Mallick et al. [31] to improve optimization and
rasterization speeds.

<!-- page 9 -->
Fig. 10: Cache amortization behavior example from the FullBody
dataset. NEE* denotes NEE times with our ray termination heuristic.
At low sample counts, our method performs better than the baseline
NEE path tracer. As SPP increase, the time to generate the cache
buffers stays constant, and the overall runtime for our method grows
more slowly than both baseline approaches.
One interesting aspect of the cache is that, unlike NRC [34],
its runtime overhead is constant in the number of samples
taken per pixel. So, while increasing SPP generally diminishes
the returns of a cache as more unbiased samples are collected,
we can still benefit from its use. Each sample can access the
same cache state to improve sample quality and shorten paths.
This results in an overall performance gain as each sample is
generated faster due to early termination, while the cache state
only needs to be rendered once per frame, regardless of SPP.
We show a runtime curve from one of our datasets in Figure 10,
which outlines how the cache usage amortizes over the number
of samples.
5
Limitations and Future Directions
In the following, we discuss the limitations of our method and
highlight areas for potential future work.
Multi-Pass Rendering and Runtime Optimizations.
Our implementation of the cache uses different sets of Gaussians
for each cache level. These levels are entirely independent of
each other. This simplifies the implementation and usage of the
cache but leads to several inefficiencies in the rasterization and
optimization pipeline, as each of these stages needs to be exe-
cuted separately for each cache level. A possible optimization
and extension of this work could be to develop a joint raster-
ization and optimization pipeline that can jointly render and
optimize multiple sets of Gaussians while still producing their
independent contributions and gradients for use in the caching
application. Furthermore, at the time of writing, differentiable
3D Gaussian splatting is still a relatively new topic, with ongo-
ing research into improving storage demands as well as splatting
and optimization times. One example of such developments is
a recent work by Mallick et al. [31], who present multiple times
faster optimization speeds at similar visual quality. As more
research is published on these topics, we anticipate the value of
our cache design to further increase.
Cache Size.
During rendering and optimization, the
cache resides entirely on the GPU. One downside of Gaussian-
splatting-based approaches is the relatively high memory re-
quirement compared to other scene representation techniques.
This can negatively impact application performance as the
caching pipeline might take up a non-negligible fraction of
GPU memory during runtime. There have been several recent
attempts to cull the memory footprint of Gaussian splatting
applications [25,31,35,40]. Extensions of our approach could
include such cache compression and reduction methods to create
a smaller and more efficient cache.
Adaptivity Versus Consistency. For a radiance cache to
be effective, it needs to react quickly to changes in the scene
by evicting or forgetting past states and adapting to new ones.
At the same time, if conditions are static, we want the cache to
provide consistent data to ensure image quality and temporal
stability between frames. In this work, we develop an adaptive
learning rate scheduler that is tied to the renderer and user
interaction patterns to determine the optimal learning rate at
any given point. For our purposes, this constitutes an adequate
trade-off between the adaptability and consistency of our cache.
However, this method is specifically tailored to our scientific
visualization use case and might not be adequate for other
applications. In such cases, there is a need to develop more
sophisticated methods to establish a balance between adaptabil-
ity and consistency in the cache. Furthermore, while our design
is largely agnostic to transfer function changes, it requires re-
initialization for cases that alter the overall morphology of the
visible volume. Future work could address refitting strategies for
such cases. Another consequence of the cache’s high adaptivity
is its short-term memory. If a viewport is not covered for some
time, the Gaussians encoding the radiance for this viewport
might be re-used to encode more recent views, thus “forgetting”
views that have not been active for some time and preventing a
global convergence of the cache in the long term. While we did
not observe unrecoverable cases of view-dependent overfitting,
this limitation requires continuous training while scene or view-
port changes are expected. Future work could address this issue
by focusing optimizations on Gaussians that are closest and
most relevant to the current viewport, potentially using spatial
subdivision and nearest-neighbor methods to find optimization
candidates to facilitate longer-term convergence of the cache.
Bias.
As with most radiance caching approaches, using
our cache results in biased rendering results. We achieve very
high image quality this way, however there are no guaranteed
error bounds for the error associated with bias.
This is a
known limitation of caching approaches. In our experiments,
we observed tolerable amounts of bias-related errors, which
speaks to our cache’s reconstruction quality. Nevertheless, if
unbiased results are a requirement, our method is not suitable.
In those cases, users might fare better employing advanced
sampling methods like ReSTIR [4] or Volume ReSTIR [29].
Self-Training and Cache Design. Currently, our cache is
trained in path space purely on unbiased path samples obtained
during rendering. This limits training data to fully concluded
paths.
Future work could explore the potential of reusing
cached entries for paths ˆn < n by demodulating them with the
throughput σi and sampling probability Tr of prior segments to
generate samples for earlier cache levels in the spirit of Müller et
al. [34]. Furthermore, recent advances in ray tracing Gaussian
representations [5,33,54] allow for arbitrary ray queries which
can enable world-space cache designs similar to NRC [34].
6
Conclusion
We introduce a novel method for radiance caching using path-
space radiance samples in combination with a multi-level hier-
archy of Gaussians. The properties of Monte Carlo noise allow
us to learn sub-space path radiance from noisy inputs using gra-
dient descent. Our evaluation results show that GSCache helps
volume path tracers converge faster and produce higher-quality
results in a shorter amount of time. At the same time, our
framework is easy to integrate into existing applications due to
its reliance on path-space radiance, unlike other popular caching
methods, which usually require extensive information about the
sampling process, intersection metadata, and more. Caching
radiance in path-space also has the advantage of being agnostic
to the characteristics of the scene. It can handle surfaces just as
easily as volumetric data and other representations like signed
distance fields or implicit neural representations. With this
work, we hope to encourage further research into the use of
Gaussian splatting for scientific visualization applications.

<!-- page 10 -->
Acknowledgments
This research is supported in part by the U.S. National Science
Foundation with grant III-2427770 and the Intel oneAPI Center
of Excellence.
References
[1] D. Bauer, Q. Wu, and K.-L. Ma. FoVolNet: Fast volume ren-
dering using foveated deep neural networks. IEEE Transactions
on Visualization and Computer Graphics, 29(1):515–525, 2023.
doi: 10.1109/TVCG.2022.3209498 3
[2] D. Bauer, Q. Wu, and K.-L. Ma. Photon field networks for
dynamic real-time volumetric global illumination. IEEE Trans-
actions on Visualization and Computer Graphics, 30(1):975–985,
2024. doi: 10.1109/TVCG.2023.3327107 2, 3
[3] P. Bekaert, P. Slussalek, R. Cools, V. Havran, and H.-P. Seidel. A
custom designed density estimator for light transport. Technical
Report, Max Planck Institut für Informatik, (4), 2003. 2, 3
[4] B. Bitterli, C. Wyman, M. Pharr, P. Shirley, A. Lefohn, and
W. Jarosz. Spatiotemporal reservoir resampling for real-time
ray tracing with dynamic direct lighting. ACM Transactions
on Graphics, 39(4):148:1–148:17, article no. 148, 2020. doi: 10.
1145/3386569.3392481 1, 9
[5] J. Condor, S. Speierer, L. Bode, A. Bozic, S. Green, P. Didyk,
and A. Jarabo. Don’t splat your gaussians: Volumetric ray-
traced primitives for modeling and rendering scattering and
emissive media. ACM Transactions on Graphics, 44(1):10:1–
10:17, article no. 10, 2025. doi: 10.1145/3711853 9
[6] E. Dappa, K. Higashigaito, J. Fornaro, S. Leschka, S. Wilder-
muth, and H. Alkadhi. Cinematic rendering–an alternative to
volume rendering for 3d computed tomography imaging. Insights
Into Imaging, 7(6):849–856, 2016. doi: 10.1007/s13244-016-0518
-1 2
[7] D. Engel and T. Ropinski. Deep volumetric ambient occlusion.
IEEE Transactions on Visualization and Computer Graphics,
27(2):1268–1278, 2021. doi: 10.1109/TVCG.2020.3030344 3
[8] Facebook Inc. PyTorch. https://pytorch.org/, 2024. [Online;
accessed 01-November-2024]. 6
[9] H. Gadirov, Q. Wu, D. Bauer, K.-L. Ma, J. B. Roerdink, and
S. Frey. Hyperflint: Hypernetwork-based flow estimation and
temporal interpolation for scientific ensemble visualization. Com-
puter Graphics Forum, 44(3):e70134, 2025. doi: 10.1111/cgf.
70134 3
[10] S. Hadadan, S. Chen, and M. Zwicker. Neural radiosity. ACM
Transactions on Graphics, 40(6):236:1–236:11, article no. 236,
2021. doi: 10.1145/3478513.3480569 1, 2, 3
[11] J. Han and C. Wang. Coordnet: Data generation and visualiza-
tion generation for time-varying volumes via a coordinate-based
neural network. IEEE Transactions on Visualization and Com-
puter Graphics, 29(12):4951–4963, 2023. doi: 10.1109/TVCG.
2022.3197203 3
[12] W. Jakob, C. Regg, and W. Jarosz. Progressive Expectation–
Maximization for hierarchical volumetric photon mapping. Com-
puter Graphics Forum (Proceedings of EGSR), 30(4), 2011. doi:
10.1111/j.1467-8659.2011.01988.x 2
[13] W. Jarosz, C. Donner, M. Zwicker, and H. W. Jensen. Radi-
ance caching for participating media. ACM Transactions on
Graphics (Presented at SIGGRAPH), 27(1):7:1–7:11, 2008. doi:
10/cwnw78 2, 3
[14] D. Jönsson, J. Kronander, T. Ropinski, and A. Ynnerman. His-
torygrams: Enabling interactive global illumination in direct
volume rendering using photon mapping. IEEE Transactions on
Visualization and Computer Graphics, 18(12):2364–2371, 2012.
doi: 10.1109/TVCG.2012.232 2
[15] D. Jönsson, E. Sundén, A. Ynnerman, and T. Ropinski. A survey
of volumetric illumination techniques for interactive volume
rendering. Computer Graphics Forum, 33(1):27–51, 2014. doi:
10.1111/cgf.12252 2
[16] L. Kandlbinder, A. Dittebrandt, A. Schipek, and C. Dachsbacher.
Optimizing path termination for radiance caching through ex-
plicit variance trading. Proceedings of the ACM on Computer
Graphics and Interactive Techniques, 7(3):33:1–33:19, article no.
33, 2024. doi: 10.1145/3675381 2, 3
[17] A. Keller, K. Dahm, and N. Binder. Path space filtering. In
ACM SIGGRAPH 2014 Talks, SIGGRAPH ’14, article no. 68,
pp. 68:1–68:1. ACM, New York, 2014. doi: 10.1145/2614106.
2614149 2
[18] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. 3d
gaussian splatting for real-time radiance field rendering. ACM
Transactions on Graphics, 42(4):139:1–139:14, article no. 139,
2023. doi: 10.1145/3592433 3, 5, 6, 11
[19] R. Khlebnikov, P. Voglreiter, M. Steinberger, B. Kainz, and
D. Schmalstieg. Parallel irradiance caching for interactive monte-
carlo direct volume rendering.
Computer Graphics Forum,
33(3):61–70, 2014. doi: 10.1111/cgf.12362 2
[20] D. P. Kingma and J. Ba.
Adam: A method for stochastic
optimization. In Y. Bengio and Y. LeCun, eds., 3rd International
Conference on Learning Representations, ICLR. San Diego, 2015.
doi: 10.48550/arXiv.1412.6980 11
[21] J. Kniss, S. Premoze, C. Hansen, and D. Ebert. Interactive
translucent volume rendering and procedural modeling. In IEEE
Visualization, pp. 109–116. IEEE, 2002. doi: 10.1109/VISUAL.
2002.1183764 2
[22] J. Kniss, S. Premoze, C. Hansen, P. Shirley, and A. McPherson.
A model for volume lighting and modeling. IEEE Transactions
on Visualization and Computer Graphics, 9(2):150–162, 2003.
doi: 10.1109/TVCG.2003.1196003 2
[23] J. Krivanek, P. Gautron, S. Pattanaik, and K. Bouatouch. Ra-
diance caching for efficient global illumination computation.
IEEE Transactions on Visualization and Computer Graphics,
11(5):550–561, 2005. doi: 10.1109/TVCG.2005.83 1, 2, 3
[24] T. Kroes, F. H. Post, and C. P. Botha. Exposure render: An
interactive photo-realistic volume rendering framework. PLOS
ONE, 7(7):1–10, 2012. doi: 10.1371/journal.pone.0038586 2
[25] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park. Compact
3d gaussian representation for radiance field. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 21719–21728, 2024. 9
[26] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras,
M. Aittala, and T. Aila. Noise2Noise: Learning image restoration
without clean data. In J. Dy and A. Krause, eds., Proceedings of
the 35th International Conference on Machine Learning, vol. 80
of Proceedings of Machine Learning Research, pp. 2965–2974.
PMLR, 2018. 5, 6
[27] M. Levoy. Display of surfaces from volume data. IEEE Computer
Graphics and Applications, 8(3):29–37, 1988. doi: 10.1109/38.
511 2
[28] D. Lin*, M. Kettunen*, B. Bitterli, J. Pantaleoni, C. Yuksel,
and C. Wyman. Generalized resampled importance sampling:
Foundations of restir. ACM Transactions on Graphics (Pro-
ceedings of SIGGRAPH 2022), 41(4):75:1–75:23, article no. 75,
2022. (*Joint First Authors). doi: 10.1145/3528223.3530158 1
[29] D. Lin, C. Wyman, and C. Yuksel. Fast volume rendering with
spatiotemporal reservoir resampling.
ACM Transactions on
Graphics (Proceedings of SIGGRAPH Asia 2021), 40(6):278:1–
278:18, article no. 278, 2021. doi: 10.1145/3478513.3480499 1,
9
[30] I. Loshchilov and F. Hutter. Decoupled weight decay regulariza-
tion. In International Conference on Learning Representations,
2017. 6, 11
[31] S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco,
and F. De La Torre. Taming 3dgs: High-quality radiance fields
with limited resources. In SIGGRAPH Asia 2024 Conference
Papers, SA ’24, article no. 2, pp. 2:1–2:11. ACM, New York,
2024. doi: 10.1145/3680528.3687694 5, 8, 9
[32] N. Max. Optical models for direct volume rendering. IEEE
Transactions on Visualization and Computer Graphics, 1(2):99–
108, 1995. doi: 10.1109/2945.468400 2
[33] N. Moenne-Loccoz, A. Mirzaei, O. Perel, R. de Lutio, J. Mar-
tinez Esturo, G. State, S. Fidler, N. Sharp, and Z. Gojcic. 3d
gaussian ray tracing: Fast tracing of particle scenes.
ACM
Transactions on Graphics, 43(6):232:1–232:19, article no. 232,
2024. doi: 10.1145/3687934 9
[34] T. Müller, F. Rousselle, J. Novák, and A. Keller. Real-time
neural radiance caching for path tracing. ACM Trans. Graph.,
40(4), article no. 36, 2021. doi: 10.1145/3450626.3459812 1, 2,

<!-- page 11 -->
3, 6, 7, 8, 9
[35] S. Niedermayr, J. Stumpfegger, and R. Westermann.
Com-
pressed 3d gaussian splatting for accelerated novel view synthe-
sis. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pp. 10349–10358, 2024.
9
[36] NVIDIA Corporation. OptiX. https://developer.nvidia.com/
rtx/ray-tracing/optix, 2024. [Online; accessed 01-November-
2024]. 6
[37] Oak
Ridge
National
Lab.
Supernova
dataset.
https://www.ornl.gov/. 7
[38] Y. Ouyang, S. Liu, M. Kettunen, M. Pharr, and J. Pantaleoni.
Restir gi: Path resampling for real-time path tracing. Computer
Graphics Forum, 40(8):17–29, 2021. doi: 10.1111/cgf.14378 1
[39] B. Pan, J. Lu, H. Li, W. Chen, Y. Wang, M. Zhu, C. Yu,
and W. Chen. Differentiable design galleries: A differentiable
approach to explore the design space of transfer functions.
IEEE Transactions on Visualization and Computer Graphics,
30(1):1369–1379, 2024. doi: 10.1109/TVCG.2023.3327371 2
[40] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Dret-
takis. Reducing the memory footprint of 3d gaussian splatting.
Proceedings of the ACM on Computer Graphics and Interac-
tive Techniques, 7(1):16:1–16:17, article no. 16, 2024. doi: 10.
1145/3651282 9
[41] M. Ruiz, I. Boada, I. Viola, S. Bruckner, M. Feixas, and M. Sbert.
Obscurance-based volume rendering framework. In H.-C. Hege,
D. Laidlaw, R. Pajarola, and O. Staadt, eds., IEEE/ EG Sym-
posium on Volume and Point-Based Graphics, pp. 113–120. The
Eurographics Association, 2008. doi: 10.2312/VG/VG-PBG08/
113-120 2
[42] M. Schott, V. Pegoraro, C. Hansen, K. Boulanger, and K. Boua-
touch. A directional occlusion shading model for interactive
direct volume rendering. Computer Graphics Forum, 28(3):855–
862, 2009. doi: 10.1111/j.1467-8659.2009.01464.x 2
[43] V. Šoltészová, D. Patel, S. Bruckner, and I. Viola. A multidi-
rectional occlusion shading model for direct volume rendering.
Computer Graphics Forum, 29(3):883–891, 2010. doi: 10.1111/j.
1467-8659.2009.01695.x 2
[44] P. Stadlbauer, W. Tatzgern, J. Mueller, M. Winter, R. Sto-
janovic, A. Weinrauch, and M. Steinberger. Adaptive multi-view
radiance caching for heterogeneous participating media. Com-
puter Graphics Forum, p. e70051, 2025. doi: 10.1111/cgf.70051
2
[45] E. Veach.
Robust Monte Carlo methods for light transport
simulation. Stanford University, 1998. 3
[46] U. Šmajdek, v. Lesar, M. Marolt, and C. Bohak. Combined
volume and surface rendering with global illumination caching.
The Visual Computer, 40(4):2491–2503, 2023. doi: 10.1007/
s00371-023-02932-9 2
[47] G. J. Ward, F. M. Rubinstein, and R. D. Clear. A ray tracing
solution for diffuse interreflection. ACM SIGGRAPH Computer
Graphics, 22(4):85–92, 1988. doi: 10.1145/378456.378490 1
[48] S. Weiss, P. Hermüller, and R. Westermann. Fast neural rep-
resentations for direct volume rendering. Computer Graphics
Forum, 41(6):196–211, 2022. doi: 10.1111/cgf.14578 3
[49] S. Weiss, M. IşIk, J. Thies, and R. Westermann.
Learning
adaptive sampling and reconstruction for volume visualization.
IEEE Transactions on Visualization and Computer Graphics,
28(7):2654–2667, 2022. doi: 10.1109/TVCG.2020.3039340 3
[50] S. Weiss and R. Westermann. Differentiable direct volume render-
ing. IEEE Transactions on Visualization and Computer Graph-
ics, 28(1):562–572, 2022. doi: 10.1109/TVCG.2021.3114769 2
[51] E. R. Woodock, T. Murphy, P. J. Hemmings, and T. C. Long-
worth.
Techniques used in the GEM code for Monte Carlo
neutronics calculation in reactors and other systems of complex
geometry. In Proceedings of the conference on the application
of computing methods to reactor problems, pp. 557–579, 1965.
2, 3, 13
[52] Q. Wu, D. Bauer, Y. Chen, and K.-L. Ma. Hyperinr: A fast
and predictive hypernetwork for implicit neural representations
via knowledge distillation, 2023. 3
[53] Q. Wu, D. Bauer, M. J. Doyle, and K.-L. Ma.
Interactive
volume visualization via multi-resolution hash encoding based
neural representation. IEEE Transactions on Visualization and
Computer Graphics, 30(8):5404–5418, 2024. doi: 10.1109/TVCG
.2023.3293121 3
[54] Q. Wu, J. M. Esturo, A. Mirzaei, N. Moënne-Loccoz, and Z. Go-
jcic. 3dgut: Enabling distorted cameras and secondary rays in
gaussian splatting. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR), pp.
26036–26046, 2025. 9
[55] D. Zavorotny, Q. Wu, D. Bauer, and K.-L. Ma. From Cluster to
Desktop: A Cache-Accelerated INR framework for Interactive
Visualization of Tera-Scale Data. In G. Reina, S. Rizzi, and
C. Gueunet, eds., Eurographics Symposium on Parallel Graphics
and Visualization. The Eurographics Association, 2025. doi: 10.
2312/pgv.20251153 3
[56] Y. Zhang, Z. Dong, and K.-L. Ma. Real-time volume rendering
in dynamic lighting environments using precomputed photon
mapping. IEEE Transactions on Visualization and Computer
Graphics, 19(8):1317–1330, 2013. doi: 10.1109/TVCG.2013.17 2
A
Ablation Study
We provide an ablation study to showcase the effect of various
choices regarding the initialization and training of our cache.
In the following, we investigate how initialization size, spacing,
and scaling of Gaussians, and various training hyperparameters
influence the final quality of the images.
A.1
Cache Size
In this study, we compare the image quality of our results
when using different cache initialization sizes N. The results
in Figure 11 indicate that from N = 10k to about N = 300k
we see significant improvements in image quality. Beyond this
size, the additional gains are marginal considering the increased
processing and storage cost of higher numbers of Gaussians.
This informed our choice of N = 300k in our other experiments.
Note that N denotes the number of Gaussians in the first cache
level and that subsequent levels follow the schedule we describe
in the main document, in which each level is half the size of
its predecessor. The lower cache sizes generally produce more
washed-out looks in the final image.
This is because fewer
Gaussians are forced to represent larger areas of the image,
which reduces local quality and adaptability to finer details. At
N = 3M, this effect becomes almost unnoticeable. We note that
the exact cache size threshold needed to produce acceptable
results can vary from dataset to dataset simply due to their
size. The FullBody dataset shown in Figure 11 was one of
the larger datasets in our tests, which is why we chose it for
this ablation.
A.2
Hyperparameters
To understand how our hyperparameter choice influences final
image quality, we conduct an ablation over several key compo-
nents in our pipeline. Figure 12 summarizes our findings. In this
study, we test different combinations of hyperparameter settings,
gradually removing constraints or replacing key components
with alternatives. Specifically, we consider the removal of the
size gap and downscaling of the initial set of Gaussians that we
describe in the main document (Figure 12 (SC)). Furthermore,
we allow optimization for Gaussian covariance (CO), which
by default we disabled for training stability. Lastly, we test
how the regularization influenced training stability by replacing
AdamW [30] with a regular Adam [20] optimizer.
The results indicate that regularization has the highest impact
on final cache quality. Without it, the cache fails to adapt to
new viewports and gradients quickly explode, leaving us with
numerous artifacts that make the cache practically unusable
(see Figure 12 Ours vs. -REG). This is in contrast to Kerbl et
al.’s [18] original implementation, which worked well without
regularization. Allowing the optimization to scale Gaussians
by setting the scaling LR to a non-zero value can help offset
training instability in cases where the optimizer fails to do so

<!-- page 12 -->
Fig. 11: Comparison of the visual quality of our method when using different cache initialization sizes N as shown on the FullBody. The columns
show images captured from different runs at 1024 SPP using an initial cache size noted at the top of each column. Results show significant gains in
quality up to around N = 300k, after which additional gains become negligible.
Fig. 12: Ablation of our hyperparameter choice. The image compares results from our configuration with various modified versions of those settings.
Data was captured by first rotating the camera around the object for 256 frames, followed by a recuperation period of another 256 frames in a still
viewport. We present different combinations of settings with different choices removed from our final settings. The labels in the figure denote the
following modifications. (-SC) removes the initial size cap on Gaussians. (-CO) removes our limitation on the scaling learning rate and allows
scaling at a LR of 0.0125. Lastly, (-REG) removes the regularization during training.
(see Figure 12 -REG vs. -CO,-REG). We found this choice
to make little difference when using regularization, and it is
up to the specific application to determine the best setting.
Finally, the initial size cap and downscaling of Gaussians helped
preserve some of the finer details in the structure of the dataset
while trading this advantage for a slightly noisier look (see
Figure 12 Ours vs. -SC). Overall, the image quality with size
cap and downscaling proved to be slightly better. In summary,
we see that our choices worked well for our set of test data, but
we acknowledge that their impact on the final image quality
and training performance might differ depending on the specific
application, and we encourage users to experiment with different
settings. Aside from that, a valuable insight we derive is that
without regularization, real-time optimization on noisy data
does not seem to be a viable option.
B
Memory Footprint
We briefly mention the memory footprint of our method in the
main document. Here, we provide a more detailed analysis in
Table 3. The cache components refer to the different parameters
stored for Gaussian splats. All data is stored as 32-bit floating-
point values, which leaves further room for optimization. All
sizes are reported in megabytes.
Table 3: Breakdown of the memory footprint of our cache per component.
Here N is the size of each cache level, which is 300K, 150K, and 75K,
respectively, for our setup.
Comp.
Dim.
L0 Size
L1 Size
L2 Size
Position
N ×3
3.43 MB
1.72 MB
0.86 MB
Rotation
N ×4
4.58 MB
2.29 MB
1.14 MB
Color
N ×3
3.43 MB
1.72 MB
0.86 MB
Scale
N ×3
3.43 MB
1.72 MB
0.86 MB
Opacity
N ×1
1.14 MB
0.57 MB
0.28 MB
Total
N ×14
16.02 MB
8.01 MB
4.01 MB
C
Cache Initialization Time
The cache is initialized at the beginning of the application
by randomly sampling the volume to generate a point cloud
that follows the density distribution of the volume and the
chosen transfer function. To this end, we use the same sampling

<!-- page 13 -->
method [51] as in the main path tracer. In Table 4, we report
average initialization timings per dataset. As our analysis shows,
this process takes too much time to be repeated every frame,
but it is not so slow as to prohibit re-initialization of the cache
upon transfer function changes.
D
Image Quality
To complement the figures in the main document,
we
provide larger versions of the qualitative results in Fig-
ures 13, 14, 15, 16, 17, and 18.

<!-- page 14 -->
Table 4: Measurement of average initialization timings over five runs per dataset or a total of 30 initializations with N = 300K. The individual
timing depends on the dataset size and average volume density as mapped by the transfer function.
Dataset
Initialization Times [ms]
Average Time [ms]
Carp
315.688, 328.575, 325.539, 309.039, 317.080
319.18
FullBody
415.402, 410.654, 416.495, 442.563, 453.676
427.76
MechanicalHand
296.847, 279.348, 305.291, 286.633, 290.239
291.67
Spider
314.366, 316.603, 310.582, 321.163, 307.771
314.10
Supernova
428.015, 440.562, 438.888, 432.853, 414.349
430.93
Zebrafish
684.607, 678.135, 675.923, 685.786, 691.880
683.27
Total
-
411.15
Fig. 13: Visual quality of our method on the Carp dataset compared to the baseline path tracer. We show results for images at 1 SPP and compare
our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event estimation (NEE).
Fig. 14: Visual quality of our method on the FullBody dataset compared to the baseline path tracer. We show results for images at 1 SPP
and compare our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event
estimation (NEE).

<!-- page 15 -->
Fig. 15: Visual quality of our method on the MechanicalHand dataset compared to the baseline path tracer. We show results for images at 2
SPP and compare our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event
estimation (NEE).
Fig. 16: Visual quality of our method on the Spider dataset compared to the baseline path tracer. We show results for images at 1 SPP and
compare our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event
estimation (NEE).

<!-- page 16 -->
Fig. 17: Visual quality of our method on the Supernova dataset compared to the baseline path tracer. We show results for images at 1 SPP
and compare our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event
estimation (NEE).
Fig. 18: Visual quality of our method on the ZebraFish dataset compared to the baseline path tracer. We show results for images at 1 SPP
and compare our method (GSCache) against a baseline volume path tracer with uniform sampling (Uniform) and a version that uses next-event
estimation (NEE).
