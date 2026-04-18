<!-- page 1 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D
Gaussian Splatting under Uncertainty
DO ˘GA YILMAZ, University College London, United Kingdom
JIALIN ZHU, Baidu Research, China
DESHAN GONG, The University of Hong Kong, China
HE WANG*, University College London, United Kingdom
50% Spatial Sparsity
Temporal Sparsity @ 10 FPS 
37% Spatio-Temporal Sparsity
GraphiXS
FreeTimeGS
GraphiXS
4DGS-1
GraphiXS
4DGS-2
Fig. 1. GraphiXS outperforms existing 4DGS methods under various types of data uncertainty. Left: 50% cameras missing; Middle: 10 FPS
low-speed cameras; Right: 37% random frames missing.
We propose a new framework to systematically incorporate data uncertainty
in Gaussian Splatting. Being the new paradigm of neural rendering, Gaussian
Splatting has been investigated in many applications, with the main effort in
extending its representation, improving its optimization process, and acceler-
ating its speed. However, one orthogonal, much needed, but under-explored
area is data uncertainty. In standard 4D Gaussian Splatting, data uncertainty
can manifest as view sparsity, missing frames, camera asynchronization, etc.
So far, there has been little research to holistically incorporating various types
of data uncertainty under a single framework. To this end, we propose Graph-
ical X Splatting, or GraphiXS, a new probabilistic framework that considers
multiple types of data uncertainty, aiming for a fundamental augmentation
of the current 4D Gaussian Splatting paradigm into a probabilistic setting.
GraphiXS is general and can be instantiated with a range of primitives, e.g.
Gaussians, Student’s-t. Furthermore, GraphiXS can be used to ‘upgrade’ exist-
ing methods to accommodate data uncertainty. Through exhaustive evaluation
and comparison, we demonstrate that GraphiXS can systematically model
various uncertainties in data, outperform existing methods in many settings
where data are missing or polluted in space and time, and therefore is a major
generalization of the current 4D Gaussian Splatting research.
CCS Concepts: • Computing methodologies →Machine learning ap-
proaches; Rendering.
*Corresponding author.
Authors’ Contact Information: Do˘ga Yılmaz, University College London, London,
United Kingdom, doga.yilmaz@ucl.ac.uk; Jialin Zhu, Baidu Research, Beijing, China,
misaliet@outlook.com; Deshan Gong, The University of Hong Kong, Hong Kong, China,
deshan@hku.hk; He Wang, University College London, London, United Kingdom,
he_wang@ucl.ac.uk.
Additional Key Words and Phrases: Gaussian Splatting, Graphical Models,
Bayesian Inference
1
Introduction
As the latest paradigm of 3D reconstruction and neural rendering,
Gaussian Splatting (GS) has served as a fundamental component of
many systems [Xiang et al. 2025; Zhou et al. 2024]. At the high level,
the current research effort can be broadly categorized as extending
the representation [Hamdi et al. 2024; Liu et al. 2025b; Zhu et al.
2025], designing new optimization strategies [Kheradmand et al.
2024; Kim et al. 2025; Zhu et al. 2025], speeding inference processes
or/and scaling up the model [Feng et al. 2025; Kerbl et al. 2024;
Mallick et al. 2024]. The three lines of research have spawned new
applications in novel view synthesis [Xie et al. 2024; Yang et al.
2025], SLAM [Matsuki et al. 2024], geometric reconstruction [Dai
et al. 2024; Huang et al. 2024], etc.
One under-explored theme that is orthogonal to all the afore-
mentioned research is data uncertainty, which universally exists in
real-world applications. Taking multi-view 4DGS as an example,
most existing research focuses on learning the dynamics of Gaussian
components, with the aim of high reconstruction quality [Wu et al.
2024; Yang et al. 2024b], generalization on unseen motions [Li et al.
2024; Zhu et al. 2024], large complex 3D scenes [Xie et al. 2025], etc.
However, all of them explicitly or implicitly assume that sufficient
data can be obtained with high quality. In practice, this often means
enough cameras with views covering all angles, and good camera
arXiv:2601.19843v1  [cs.GR]  27 Jan 2026

<!-- page 2 -->
2
•
Yilmaz et al.
calibration and synchronization. We argue that such assumptions can
be too restrictive as the data collection setup can be constrained by
many factors such as security, safety, operational constraints, e.g.
cameras at traffic junctions might only cover a few angles.
Recently, some methods, including contemporaneous ones, have
started to consider certain data uncertainties in GS. This includes
probabilistic inference for online learning [Guo et al. 2025; Savant
et al. 2024; Van de Maele et al. 2024], dynamically learning moving
and static Gaussians [Deng et al. 2025; Gao et al. 2024; Wang et al.
2025a], reconstruction from limited camera views [Jeong et al. 2024;
Jiang et al. 2023; Yılmaz and Kıraç 2023]. However, the data uncer-
tainty explicitly or implicitly considered is mostly specific to one
application scenario. Therefore, it is desirable to have one unified
framework that can incorporate multiple types of data uncertainty.
We propose Graphical X Splatting (GraphiXS), a new framework
which can explicitly incorporate multiple types of data uncertainty in
4DGS. The ‘X’ in GraphiXS is not necessarily Gaussian so we use
the word ‘component’. The term ‘data uncertainty’ broadly refers to
missing camera views, sparse camera configurations (i.e. position
and orientations), missing frames from cameras, imperfectly syn-
chronized cameras, etc. Probabilistic learning is a natural solution
to data uncertainty, but the greatest challenge is to design a flexible
probabilistic framework that can model different types of uncertainty
in 4DGS. To this end, we formulate GraphiXS as a generative process
and propose a new graphical model, by introducing stochasticity into
the individual steps of 4DGS. This includes treating all the learn-
able parameters (e.g. component location) as latent variables which
are to be inferred via Maximum a Posteriori (MAP). Also, unlike
existing methods which treat only the images as observations, we
also treat the camera pose and frame time as samples of random
variables, which enables us to incorporate the uncertainty in them.
Finally, the flexibility of GraphiXS allows us to impose various prior
distributions to regulate the behaviors of the components.
We instantiate GraphiXS with different components including
Gaussians and Student’s-t to show its generality. Through exhaus-
tive evaluation under various combinations of data uncertainty, we
demonstrate that GraphiXS can outperform existing methods across
different scenes and metrics. More broadly, we demonstrate that
GraphiXS is not a specific model but a framework which can be
used to upgrade and improve existing methods. To the best of our
knowledge, this is the first probabilistic 4DGS framework targeting
multiple types of data uncertainty. Our contributions include:
• A new probabilistic framework to holistically incorporate
data uncertainty in 4DGS.
• A new graphical model which can be instantiated with differ-
ent primitives and used to ‘upgrade’ existing 4DGS methods.
• A new way of introducing stochasticity in the steps of 4DGS.
• New priors that can effectively regulate the model behaviors,
leading to effective optimization.
2
Related Work
Traditional Methods. Multi-View Stereo (MVS) [Seitz et al. 2006]
is well studied before deep learning. Software such as Colmap [Schön-
berger et al. 2016] has been widely utilized in 3D reconstruction. It
reconstructs the geometrically consistent points between images and
restores camera poses by calculating corresponding features in im-
ages from multiple perspectives. Denser and more precise geometries
can be obtained using the Structure from Motion (SFM) [Ullman
1979]. Meanwhile, additional data (e.g. depth) captured from Time
of Flight (ToF) or Light Detection And Ranging (Lidar) sensors can
be accessed from some other methods such as SLAM [Bailey and
Durrant-Whyte 2006; Durrant-Whyte and Bailey 2006] and Kinect-
Fusion [Newcombe et al. 2011] for 3D reconstruction.
Learning-based Reconstruction Method. For a single scene, there
are two main cornerstone methods in this sub-area: Neural Radiance
Field (NeRF) [Mildenhall et al. 2021] and 3D Gaussian Splatting
(3DGS) [Kerbl et al. 2023]. NeRF utilizes a neural network to implic-
itly learn the 3D radiance field. It can achieve novel view synthesis
by querying different points’ color and density values from the neu-
ral network. NeRF accomplishes good reconstruction results, but
its rendering efficiency is low, making it unsuitable for real-time
rendering tasks. Comparatively, 3DGS can achieve real-time render-
ing, but requires more memory compared with NeRF. 3DGS uses
3D Gaussians as components in space. To optimize their attributes
(means, covariances, colors, etc.), it uses a rasterization method called
Splatting [Zwicker et al. 2002] to obtain rendering results from dif-
ferent perspectives. Many subsequent methods are then proposed to
improve 3DGS. Among them, some attempt to improve 3DGS in
the fundamental paradigm, e.g. using different primitives other than
Gaussian including SSS [Zhu et al. 2025], DBS [Liu et al. 2025b],
and 2DGS [Huang et al. 2024], while others try to improve the train-
ing processing and adaptive density control in vanilla 3DGS, such as
sampling-based [Kheradmand et al. 2024; Kim et al. 2025; Zhu et al.
2025] methods, elevating rendering quality to a new level.
Dynamic Reconstruction. Most traditional methods either per-
form a frame-by-frame reconstruction and lack motion continuity,
or require calculating optical flow maps to simulate dynamics but
cannot reconstruct dense scene flows. In comparison, dynamic re-
construction based on NeRF and 3DGS yields better results. D-
NeRF [Pumarola et al. 2021] extend NeRFs to dynamic scenes by
introducing an MLP to learn the implicit deformation field in every
time interval for the motion. Because of the good learning capability
of the implicit representation of NeRF, most NeRF-based dynamic re-
construction methods adopt concepts that are similar to D-NeRF. On
the other hand, there are currently two main threads for dynamic
reconstruction using 3DGS-based methods. The first is learning
the trajectories of explicit primitives at different times. Deformable
3DGS [Yang et al. 2024a] and 4DGS-2 [Wu et al. 2024] follow
the idea of D-NeRF, using neural networks/Tri-planes/HexPlanes to
learn the motion and deformation of 3D Gaussians. MotionGS [Zhu
et al. 2024], SplineGS [Park et al. 2025], and FreetimeGS [Wang
et al. 2025b], on the other hand, simulate the dynamics of 3D Gaus-
sians in space through optical flow, spline, and linear flow. The
second thread involves building higher-dimensional primitives based
on the properties of explicit primitives in the 3DGS method. Then,
these properties of primitives at different times in 3D space can be
calculated by marginalizing the time dimension. Research such as
4DGS-1 [Yang et al. 2024b], 7DGS [Gao et al. 2025], and UBS [Liu
et al. 2025a] are all under this direction.

<!-- page 3 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty
•
3
Uncertainty in Reconstruction. Existing works such as SGS [Sa-
vant et al. 2024] and USPLAT4D [Guo et al. 2025] have attempted
to improve reconstruction quality by estimating the uncertainty of
primitives in Gaussian Splatting. However, only few methods take
into account the inherent uncertainty of the training data besides the
primitive uncertainty. Yang et al. [2024a] argued that inaccuracies
in pose estimation can cause spatial jitter between frames. Research
including LongSplat [Lin et al. 2025], DG-SLAM [Xu et al. 2024]
and GS-CPR [Liu et al. 2024] adjusts camera poses during the train-
ing process. Besides, Bui et al. [2025] pointed out that the temporal
uncertainty should not be neglected, especially for the photon inte-
gration process within the physical shutter time. Nevertheless, there
is no work modeling further data uncertainty such as asynchronous
cameras and performing probabilistic modeling of multiple types of
uncertainty for 4DGS.
3
Methodology
3.1
Preliminaries
3.1.1
3D/4DGS as a mixture model. GS utilizes a large num-
ber of 3D Gaussian components to fit a 3D radiance field. Each
component is represented by
𝐺(𝑥) = 𝑒−1
2 (𝑥−𝜇)𝑇Σ−1 (𝑥−𝜇)
(1)
where 𝜇and Σ are the location and shape (with truncation). In ad-
dition, each component has additional attributes, including opacity
𝑜, and color 𝑠which is based on spherical harmonics 𝑠ℎ. During
rendering, each 3D Gaussian is transformed into a 2D Gaussian on
the image plane. The rendering is then computed by:
𝐶(𝑑) =
𝑁
∑︁
𝑖=1
𝑠𝑖𝑜𝑖𝐺2𝐷
𝑖
(𝑑)
𝑖−1
Ö
𝑗=1
(1 −𝑜𝑗𝐺2𝐷
𝑗(𝑑)).
(2)
where 𝐶(𝑑) is the final color of the 𝑑th pixel, 𝑁is the number of the
Gaussians that intersect with the ray cast from the pixel. All Gaussian
parameters are learned from the 2D images.
Essentially 3D/4DGS can be seen as learning a (unnormalized)
mixture model with Gaussians [Zhu et al. 2025]:
𝐹(𝑥) =
∑︁
𝑤𝑖𝐺𝑖(𝑥)
(3)
where 𝑤𝑖is determined by 𝑜𝑖, 𝑠𝑖and the rendering process. Our
GraphiXS generalizes this concept by a new graphical model and a
corresponding generative process based on the mixture model.
3.1.2
Graphical model. Graphical Model is a probabilistic model
which uses a graph to express the conditional dependencies of ran-
dom variables. It allows structured dependencies to be introduced
among random variables to describe a data generation process. Then
the knowledge of this process can be used as inductive bias for model
design. This is particularly suitable for 4DGS since it describes a
multi-step process of rendering (Eq.1-3) and each step involves sev-
eral quantities which can be treated as random variables. Also, Graph-
ical Model naturally enables Bayesian inference. This is achieved
by factorizing the joint probability of all variables into a series of
conditional probabilities based on the graph, each of which can be
independently modeled and evaluated. For a complex process like
4DGS, as shown later, this provides the flexibility for us to design
separate priors to regulate overall model behaviors. Below, we first
introduce the graphical model and its corresponding generative pro-
cess, followed by model decomposition. Then we explain how to
model each of the terms in the model. Lastly, we derive the final loss
function for training.
3.2
GraphiXS as a Graphical Model
Notations. Given videos in a multi-view setting, we aim to re-
construct the 4D radiance field. We define random variables 𝐶for
the cameras pose, 𝑇for frame time, 𝐼for video frame with pixels
𝑋. The data consists of 𝑀camera poses, each recording 𝐾frames,
giving a total of 𝑀× 𝐾frames and each frame with 𝐷pixels. We use
superscripts for time, superscripts with brackets for derivatives, and
subscripts for other indices, e.g. 𝑋𝑡
𝑐,𝑑indicates the 𝑑th pixel of the
image 𝐼𝑡
𝑐from camera 𝑐at time 𝑡.
Our new GraphiXS is a graphical model (Fig. 2) which represents
a generative process from a (unnormalized) mixture model with
a potentially infinite number of components 𝐹= Í𝑁
𝑖=1 𝛿𝜃𝑖, where
𝑁→+∞and 𝛿𝜃𝑖is the 𝑖th component parameterized by 𝜃𝑖. The
generative process is then described as follows:
(1) Given the mixture model 𝐹, a camera 𝑐∼𝐶and a time 𝑡∼𝑇,
sample a subset of 𝐹which is called {𝛼𝑖} with L components,
for an image 𝐼𝑡
𝑐.
(2) For every 𝑋𝑡
𝑐,𝑑∈𝐼𝑡
𝑐indexed by 𝑐, 𝑡, and 𝑑, sample a ray 𝑅𝑡
𝑐,𝑑.
(3) Given {𝛼𝑖} and 𝑅𝑡
𝑐,𝑑, sample a subset of {𝛼𝑖} which is called
{𝛽𝑗} with A components.
(4) Finally, generate the color for 𝑋𝑡
𝑐,𝑡based on {𝛽𝑗} and 𝑅𝑡
𝑐,𝑑.
Figure 2 describes most of the existing GS work where some steps
are realized as rule-based deterministic processes. For instance, the
original 3DGS is special case of GraphiXS without 𝑇and with 𝛿𝜃𝑖
being Gaussian. Then the last two steps correspond to a per-pixel
intersection test between a ray and the Gaussians, and rasterization
process. The subset {𝛼𝑖} corresponds to the Gaussians that are pro-
jected onto the image plane of camera 𝑐at time 𝑡. The subset {𝛽𝑗}
corresponds to the Gaussians hit by the ray 𝑅𝑡
𝑐,𝑑cast for a pixel.
The key learnable parameter is 𝜃. It is ideal to make a full Bayesian
inference on the posterior distribution of 𝜃, since there is more than
one set of 𝜃s which can provide good reconstruction. In practice, this
is extremely challenging due to the large number of 𝜃. Models such
as Hierarchical Dirichlet Processes could describe our generative
process by assuming conjugate priors (e.g. Dirichlet-Multinomial),
so that intermediate latent variables such as 𝛼, 𝛽can be marginalized.
However, some steps of the generative process are dictated by light
transport which cannot be easily described by probabilistic distribu-
tions. Furthermore, even if we forcefully used a full probabilistic
Fig. 2. A graphical model for GraphiXS. The colors correspond to the
4 steps of the generative process (1: red, 2: green, 3: yellow, 4: purple)

<!-- page 4 -->
4
•
Yilmaz et al.
model, the optimization would involve intensive sampling or large-
scale variational inference [Van de Maele et al. 2024], both being
prohibitively slow in the presence of millions of 𝜃s. Therefore, we
choose Maximum a Posteriori (MAP):
arg max
𝜃
𝑃(𝑋, | 𝑅, 𝛽, 𝛼,𝐶,𝑇,𝛿𝜃)𝑃(𝛿𝜃)
(4)
where we assume uninformative priors for 𝐶and 𝑇. Despite not di-
rectly modeling their priors, we do implicitly consider their influence
on the distribution of the components as random variables, explained
later. We first factorize the likelihood following the graphical model:
𝑃(𝑋| 𝑅, 𝛽,𝛼,𝐶,𝑇,𝛿𝜃) = 𝑃(𝑋| •)
= 𝑃(𝑋| 𝛽, 𝑅)𝑃(𝛽| 𝛼, 𝑅)𝑃(𝑅| 𝐶,𝑇)𝑃(𝛼| 𝛿𝜃,𝐶,𝑇), (5)
where we use • to represent variables in the condition for brevity.
Also, among the possible options for instantiating 𝛿in GraphiXS,
e.g. Gaussians, Student’s-t, Beta, etc., we assume the component is
a distribution parameterized by basic attributes including mean 𝜇,
variance Σ, color 𝑠, opacity 𝑜, and other dynamics related attributes
introduced later. Although this assumption excludes free-form pa-
rameterizations such as shapes [Held et al. 2025], it is still valid
for many GS frameworks. Below, we give the key equations of our
instantiations of different distributions in Eq. (5) and provide the
details in the supplementary material (SM).
3.3
Probabilistic Image Reconstruction
𝑃(𝑋| •) represents the image generative process with 4 distributions.
As mentioned, some of the steps are dictated by the light transport
and are deterministic. So we realize them as deterministic processes
following existing practice and introduce stochasticity in the others:
𝑃(𝑋| •) = 𝑃(𝑋| 𝛽, 𝑅)
|       {z       }
rasterization
𝑃(𝛽| 𝛼, 𝑅)
|       {z       }
per-pixel component
𝑃(𝑅| 𝐶,𝑇)
|       {z       }
per-pixel ray
𝑃(𝛼| 𝛿𝜃,𝐶,𝑇)
|           {z           }
per-image component
(6)
Following 3DGS [Kerbl et al. 2023], we realize 𝑃(𝑋| 𝛽, 𝑅) as
rasterization:
𝑅𝑎𝑠(𝑋; 𝛽, 𝑅) =
𝑁
∑︁
𝑖=1
𝜆𝑖𝛽2𝐷𝑖(𝑋), 𝜆𝑖= 𝑠𝑖𝑜𝑖
𝑖−1
Ö
𝑗=1
(1 −𝑜𝑗𝛽2𝐷𝑗(𝑋))
(7)
where 𝛽2𝐷is a component projected onto the image space. Similarly,
𝑃(𝛽| 𝛼, 𝑅) = 𝐼𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡(𝑅, 𝛼) is the per-pixel intersection test to
identify the relevant components 𝛽. 𝑃(𝑅| 𝐶,𝑇) = 𝑅𝑎𝑦𝐶𝑎𝑠𝑡𝑖𝑛𝑔(𝐶,𝑇)
is ray casting which describes a ray 𝑅from a camera 𝑐into the space
at time 𝑡. Combining the first three steps gives:
𝑋= 𝑅𝑎𝑠(𝑋; 𝐼𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡(𝑅, 𝛼), 𝑅𝑎𝑦𝐶𝑎𝑠𝑡𝑖𝑛𝑔(𝐶,𝑇)) = 𝑅𝑎𝑠(𝑋; •),
(8)
which is not a distribution and therefore cannot be directly used for
MAP. So we use an energy-based distribution [Zhu et al. 2025]:
𝑃(𝐼| 𝛽, 𝑅, 𝛼,𝐶,𝑇) ∝𝑒𝑥𝑝(−
𝑀
∑︁
𝑐
𝐾
∑︁
𝑡
𝐿𝑖𝑚𝑔(𝐼𝑡
𝑐))
𝐿𝑖𝑚𝑔= (1 −𝜖𝐷−𝑆𝑆𝐼𝑀)𝐿1 + 𝜖𝐷−𝑆𝑆𝐼𝑀𝐿𝐷−𝑆𝑆𝐼𝑀
+ 𝜖𝑜
∑︁
𝑖
||𝑜𝑖||1 + 𝜖Σ
∑︁
𝑖
∑︁
𝑗
||
√︁
𝜆𝑖,𝑗||1
(9)
where 𝐿1 and 𝐿𝐷−𝑆𝑆𝐼𝑀are the 𝐿1 norm and the structural similarity
loss between the reconstructed image (by Eq. (8)) and the ground-
truth. 𝜆s are the eigenvalues of Σ. The regularization applied to the
opacity ensures that the opacity is big only when a component is
absolutely needed. The regularization on 𝜆ensures the model uses
components as spiky as possible (i.e. small variances). Together, the
regularization terms minimize the needed number of components.
Next, since there are theoretically an infinite number of config-
urations of 𝜃s which can reconstruct the same radiance field, we
argue that it is important to impose preferences on well-behaved
𝜃s. We impose this preference as stochasticity where well-behaved
components have higher probabilities, via:
𝑃(𝛼| 𝛿𝜃,𝐶,𝑇) = 𝐸{𝑃(𝑟𝑡
𝑐,𝑑|𝛼)} ≈
Í𝑀
𝑐
Í𝐾
𝑡
Í𝐷
𝑑𝑃(𝑟𝑡
𝑐,𝑑|𝛼)
Í𝑁
𝑝
Í𝑀
𝑐
Í𝐾
𝑡
Í𝐷
𝑑𝑃(𝑟𝑡
𝑐,𝑑|𝛿𝜃𝑝)
(10)
where 𝑃(𝑟𝑡
𝑐,𝑑|𝛼) and 𝑃(𝑟𝑡
𝑐,𝑑|𝛿𝜃𝑝) are the likelihoods of a ray 𝑟𝑡
𝑐,𝑑from
camera 𝑐at time 𝑡with respect to a component 𝛼and 𝛿𝜃𝑝. 𝑀, 𝐾, 𝐷,
𝑁are the total number of cameras, frames, pixels per image, and
components. 𝑃(𝑟𝑡
𝑐,𝑑|𝛿𝜃𝑝) is the soft visibility of 𝛿𝜃𝑝to a pixel 𝑥𝑡
𝑐,𝑑.
When 𝛿𝜃𝑝has low variance and high 𝑃(𝑟𝑡
𝑐,𝑑|𝛿𝜃𝑝), 𝛿𝜃𝑝is highly visible
to 𝑋𝑡
𝑐,𝑑. Overall, in Eq. (10), the numerator is the soft visibility of
component 𝛼to all pixels of all images across all times, while the
denominator is the sum of all the soft visibility of all components.
Therefore, we call 𝑃(𝛼| 𝛿𝜃,𝐶,𝑇) the component confidence. A
component with high confidence is more visible to all pixels in all
frames relative to other components.
Aside from the visibility of components to cameras in time, con-
versely, 𝑃(𝛼| 𝛿𝜃,𝐶,𝑇) also implicitly considers the influence of the
distributions of 𝐶and 𝑇on the components. This is important for
accommodating sparse views or missing frames. Maximizing this
probabilistic distribution means placing components with high prob-
abilities in the overlapped visible regions of multiple cameras and
times. If the data is missing from any camera or time, the correctly
placed components will be able to make good ‘guesses’ of the 4D
radiance field for the missing cameras and frame times, hence robust
to these uncertainties.
3.4
Prior for Component Parameters
After the likelihood function, we model the prior 𝑃(𝛿𝜃) by modeling
the distribution of the parameter 𝜃. We assume our components move
in space and time, and parameterize 𝑃(𝜃) to capture their dynamics,
assuming 𝜃(𝑡) is function of 𝑡:
𝑃(𝜃(𝑡)) = 𝑃(𝜃(1))
𝐾
Ö
𝑡=2
𝑃(𝜃(𝑡) | 𝜃(𝑡−1))
(11)
The full specification of 𝜃= {𝜇, Σ,𝑜,𝑠ℎ,𝑔,𝑢, 𝑣,𝑎, 𝑗,𝑠} includes a set
of learnable parameters of the components, where 𝜇, Σ are the mean
and covariance. 𝑜is the base opacity. 𝑠ℎdenotes its color represented
through spherical harmonics. In time, a component can appear at any
time 𝑔and last for a period of time 𝑢. In addition, we also introduce
other dynamics related variables 𝑣, 𝑎, 𝑗, 𝑠where are detailed later.
We do not treat all variables in 𝜃as functions of time explicitly.
Instead, we explicitly model 𝜇as it is the main variable governing the
location of the components. We also make 𝑠ℎdependent on 𝜇. The
rest is learned independently. Note that the current 𝜃specification is

<!-- page 5 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty
•
5
the minimal set of variables to model component dynamics. They are
broadly shared in multiple types of components, e.g. Gaussian, Beta,
Student’s-t. There might be other parameters for certain choices of
components, such as the control parameter in Student’s-t. In this case,
these additional parameters are learned independently.
First, we model 𝜇(𝑡) as Brownian motions:
𝑑𝜇
𝑑𝑡= 𝑓(𝜇,𝑡) + N (0,𝜖2I)
(12)
where N is a Normal distribution with standard deviation 𝜖. I is iden-
tity matrix. The reason is that we observe that the motions of objects
are arbitrary and there is even no guarantee of motion smoothness.
Therefore, it is crucial to be able to learn arbitrary and even poten-
tially discontinuous motions. Discretizing Eq. (12) in time gives:
𝜇(𝑡) = 𝜇(𝑡−1) + 𝑓(𝜇,𝑡)△𝑡+ N (0,𝜖2I)△𝑡
(13)
where the dynamics is governed by 𝑓(𝜇,𝑡). One simple way of model-
ing 𝑓(𝜇,𝑡) is to consider the velocity of the components as in existing
methods [Wu et al. 2024]. However, to also consider highly discontin-
uous motions, we learn several orders of motion derivatives, velocity
𝑣, acceleration 𝑎, jerk 𝑗and snap 𝑠:
𝑓(𝜇,𝑡)△𝑡= 𝑣(𝑡−𝜇𝑡)+ 1
2𝑎(𝑡−𝜇𝑡)2+ 1
6 𝑗(𝑡−𝜇𝑡)3+ 1
24𝑠(𝑡−𝜇𝑡)4 (14)
Besides, we also design priors on other parameters. This is because
MAP for GraphiXS involves millions of parameters. It can easily
overfit without prior knowledge to regulate the variables. First, we
impose a prior on the base opacity 𝑜following [Wang et al. 2025b]:
𝑃(𝑜) ∝1
𝑁
𝑁
∑︁
1
𝑜2 exp

−1
2
𝑡−𝑔
𝑢

𝜓(·),
(15)
where 𝑁is the total number fo components, and 𝜓(·) denotes the
evaluated response of the splatting distribution. The inputs to 𝜓(·)
depend on the chosen component and its associated parameters. For
Gaussian, 𝜓(·) is evaluated using the mean and variance, whereas for
Student’s-t we additionally include its control parameter 𝜈. Please
see the SM for details. Overall this prior penalizes excessively large
base opacity values.
Next, we also impose a prior on Σ:
𝑃(Σ𝑡
𝑝) ∝𝑒𝑥𝑝(−𝜆𝜎
1
𝑁𝑡
Nt
∑︁
𝑝
||Σ𝑡
𝑝−ˆΣ𝑡𝑝||2
𝐹)
(16)
where Σ𝑡
𝑝is the covariance of the 𝑝th component at time 𝑡, 𝜆𝜎is a
weight, ˆΣ𝑡𝑝is the mean covariance of all components at time 𝑡, || · ||𝐹
is the Frobenius norm, 𝑁𝑡is the total number of components at time
𝑡. This prior prefers small shape disparities between components.
We also place a prior on the dynamics related variables:
𝑃(𝑣𝑡
𝑝,𝑎𝑡
𝑝, 𝑗𝑡
𝑝,𝑠𝑡
𝑝) ∝𝑒𝑥𝑝(−𝜆ℎ
1
Pt
Pt
∑︁
𝑝
√︃
| det(Σ𝑡𝑝)|
(||𝑣𝑡
𝑝||2 + ||𝑎𝑡
𝑝||2 + ||𝑗𝑡
𝑝||2 + ||𝑠𝑡
𝑝||2)
(17)
which prefers slowly and smoothly moving components. More im-
portantly, it is weighted by the volume proxy of the component
| det(Σ𝑡
𝑝)|, so that the larger the component is, the slower and smoother
its motion should be. This is based on the observation that larger
components cover a large area in space. These tend to be in the static
background, so they should not move too often and too quickly.
Finally, combining Eq. (12-17), Eq. (11) becomes:
𝑃(𝜃(𝑡)) = 𝑃(𝑜)𝑃(𝜇(1))
𝐾
Ö
𝑡=2
N (𝜇(𝑡−1) + 𝑓(𝜇,𝑡),𝜖2I)
𝐾
Ö
𝑡=1
𝑃(𝑣𝑡,𝑎𝑡, 𝑗𝑡,𝑠𝑡)𝑃(Σ𝑡)
(18)
with the rest parameters learned directly without stochasticity. 𝑃(𝜃(1))
is realized by Eq. (9) for the first frame.
3.5
Objective Function and Optimization
Finally, with Eq. 9-18, we have all the elements for MAP (Eq. (4)):
arg max
𝜃
𝑙𝑜𝑔𝑃(𝑋| 𝑅, 𝛽, 𝛼,𝐶,𝑇,𝜃)𝑃(𝜃)
⇔arg min
𝜃
−𝑙𝑜𝑔[𝑃(𝑋| 𝑅, 𝛽, 𝛼,𝐶,𝑇,𝜃)𝑃(𝜃)]
(19)
where the final loss function is:
Lfull =
L𝑖𝑚𝑔
|{z}
−𝑙𝑜𝑔𝑃(𝐼|•)
+
L𝛼
|{z}
−𝑙𝑜𝑔𝑃(𝛼|•)
+
L𝜃
|{z}
−𝑙𝑜𝑔𝑃(𝜃)
(20)
For optimization, we use Stochastic Gradient Hamiltonian Monte
Carlo (SGHMC) [Zhu et al. 2025], which injects momentum and
controlled stochasticity into gradient updates. In addition, we design
component addition/removal sampling and initialization strategies
for GraphiXS. Details can be found in the SM.
4
Experiments
Dataset. To evaluate GraphiXS, we need to mimic different types
of data uncertainty, which requires the original dataset to have enough
cameras, views with good coverage, sufficiently high frequency, etc.
Based on the criteria, we choose the Neural 3D Video (N3DV) dataset
[Li et al. 2022]. Following prior work [Lee et al. 2024; Wang et al.
2025b; Wu et al. 2024; Yang et al. 2024b], we down-sample all videos
to 1352×1014 for both training and evaluation. We consistently hold
out the first camera as the test view across all experimental settings.
We evaluate reconstruction quality on this view over a duration of
300 consecutive frames at 30 FPS.
Metrics. We use PSNR (Peak Signal-to-Noise Ratio), DSSIM (Dis-
similarity Structural Similarity Index Measure), and LPIPS (Learned
Perceptual Image Patch Similarity) [Zhang et al. 2018]. For LPIPS,
we report results computed using AlexNet. DSSIM is derived from
the multi-scale structural similarity (MS-SSIM) index by converting
similarity to a dissimilarity measure and scaled by a factor of 0.5.
Baseline Methods. We choose a representative set of state-of-the-
art methods as baselines including 4DGS-1 [Yang et al. 2024b],
4DGS-2 [Wu et al. 2024], Ex4DGS [Lee et al. 2024], and Free-
TimeGS [Wang et al. 2025b]. We use their official open-source im-
plementations when available, otherwise our own implementation
(for FreeTimeGS [Wang et al. 2025b]). For fair comparison, we train
all models from scratch and run them 3 times and report the average.
We color the best and second best results in green and yellow respec-
tively. We only report results for the whole dataset in the main paper
and include more detailed results and analysis in the SM.

<!-- page 6 -->
6
•
Yilmaz et al.
Table 1. Comparison under standard setting.
Method
PSNR ↑
DSSIM ↓
LPIPS ↓
4DGS-1
31.18
0.015
0.051
4DGS-2
30.52
0.019
0.060
Ex4DGS
31.71
0.015
0.050
FTGS
31.55
0.016
0.047
Ours (GraphiGS)
31.78
0.015
0.044
Ours (GraphiTS)
32.02
0.015
0.043
Uncertainty Settings. We design several settings for multiple types
of commonly seen data uncertainty. Standard Setting uses exactly the
same setting as the baseline methods, regarded as without any data
uncertainty. Sparse Views represents missing cameras. Sparse Frames
represents only low frequency cameras are used. Unsynchronized
Cameras represents a group of cameras which are not synchronized.
Faulty Cameras represents a system with random camera malfunc-
tions. Together these settings cover a wide range of possible scenarios
where data uncertainty is induced into the data.
Instantiation and Generalization. GraphiXS does not assume spe-
cific components, so we instantiate it with two components to show
its generality. The first is Student’s-t (GraphiTS) and the second is
approximate Gaussian (GraphiGS) by fixing the control parameter of
Student’s-t to a large value. In addition, we show that GraphiXS can
be used to ‘upgrade’ existing 4DGS methods.
4.1
Comparison under Standard Setting
We show the numerical results in Tab. 1. Overall, the DSSIM for
most methods are similar showing all methods capture the struc-
ture of images well in reconstruction. Both GraphiTS and GraphiGS
outperform existing methods, mainly on PSNR and LPIPS. This is
somewhat surprising, as the camera angles in N3DV are dense e.g.
little data uncertainty, so one would assume further modeling of data
uncertainty is redundant. However, the experiments demonstrate that
even with the full observations, explicitly considering data uncer-
tainty can further enhance the reconstruction quality. We show one
visual result in Fig. 3. This is a difficult scene. One example is that
one hand of the person holding a tong moves quickly from time to
time, causing visual blur. This motion involves constantly stirring the
spinach in random manner both spatially and in terms of its dynamics,
causing motion blur. So capturing the high-order dynamics is crucial.
Therefore, comparatively GraphiXS reconstructs clearer structures
of the hand and tong with details, demonstrating the dynamics of the
components are learned well.
4.2
Comparison under Sparse Views
In Sparse Views, we randomly remove 10%, 30%, and 50% of the
training cameras, causing view gaps. Table 2 shows the numerical
comparison. Overall, GraphiXS outperforms or is in par with other
methods across all metrics. Due to the dense nature of the cameras
in N3DV, a 10% reduction of the cameras does not challenge the
methods, sometimes even improve the results (e.g. PSNR in 4DGS-1,
4DGS-2 and Ex4DGS). Further analysis suggests that the difference
might be due to the randomness in the training for different methods,
which might cause bigger variances in different runs in some meth-
ods but overall give similar results to the Standard Setting. When
Table 2. Comparison at different levels of spatial sparsity.
Method
10%
30%
50%
PSNR↑DSSIM↓LPIPS↓PSNR↑DSSIM↓LPIPS↓PSNR↑DSSIM↓LPIPS↓
4DGS-1
31.37
0.015
0.049
29.11
0.024
0.058
28.54
0.027
0.068
4DGS-2
31.00
0.016
0.057
28.68
0.024
0.069
27.99
0.029
0.074
Ex4DGS
30.94
0.016
0.051
28.86
0.026
0.064
28.33
0.028
0.067
FTGS
31.38
0.016
0.047
29.04
0.026
0.063
28.06
0.031
0.072
Ours (GraphiGS) 31.73
0.016
0.044
29.35
0.025
0.058
28.60
0.026
0.066
Ours (GraphiTS)
31.62
0.016
0.046
29.41
0.024
0.061
28.56
0.027
0.067
Table 3. Comparison at different levels of temporal sparsity.
Method
20 FPS
10 FPS
PSNR↑
DSSIM↓
LPIPS↓
PSNR↑
DSSIM↓
LPIPS↓
4DGS-1
31.26
0.015
0.049
30.72
0.016
0.051
4DGS-2
30.46
0.016
0.058
31.01
0.016
0.059
Ex4DGS
31.39
0.015
0.050
31.15
0.016
0.053
FTGS
31.39
0.017
0.046
31.36
0.017
0.046
Ours (GraphiGS)
31.88
0.015
0.043
31.87
0.015
0.044
Ours (GraphiTS)
31.62
0.015
0.043
31.63
0.015
0.044
dropping 30% and 50% of the cameras, all metrics start to deteriorate,
showing Sparse Views induces major data uncertainty. Across all
three settings, GraphiXS achieves the best in 8 of 9 experiments and
the second best in 1 experiment. We show one visual result in Fig. 1
Left and more results in Fig. 4. When more cameras are missing,
it becomes more challenging for all methods where GraphiXS is
affected the least.
4.3
Comparison under Sparse Frames
In Sparse Frames, we reduce the frame rate of the training cameras
from 30 FPS to 20 and 10 FPS while still evaluating the scene at
30 FPS causing time sparsity. Table 3 shows the numerical results
for both settings. Overall GraphiXS outperforms other methods. Dif-
ferent from Sparse Views, Sparse Frames mimics low frequency
cameras which are not suitable for recording fast motions. Since mo-
tions in N3DV are generally not fast, and all methods have dedicated
parts to learn the dynamics of the components, Sparse Frames is
generally less challenging than Sparse Views.
Furthermore, the key difference between different methods stem
from how dense in time they require samples to be. However in-
dividual model behaviors do not share the same pattern. For the
baseline methods, when FPS is lower, the results become worse as
expected. The only exception is 4DGS-2 which is slightly improved.
Our speculation is that since 4DGS-2 exhaustively learns pair-wise
relationships between the 𝑥, 𝑦, 𝑧coordinates of the Gaussian mean
and time 𝑡, there is a chance their network overfits when the sample
density in time is too high. So down-sampling actually improves the
results. Last, GraphiGS does not deteriorate from Standard Setting,
GraphiTS deteriorates at 20 FPS but not further in 10 FPS. We show
one visual result in Fig. 1 Middle and more results in Fig. 5.
4.4
Comparison under Unsynchronized Cameras
In data capture, synchronizing cameras require extra hardware, soft-
ware and calibration effort. So we test if methods could work with

<!-- page 7 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty
•
7
Table 4. Comparison under different levels of Unsynchronization.
Method
10% @ 20 FPS
50% @ 20 FPS
PSNR↑
DSSIM↓
LPIPS↓
PSNR↑
DSSIM↓
LPIPS↓
4DGS-1
31.57
0.015
0.049
31.54
0.015
0.050
4DGS-2
30.68
0.016
0.059
30.92
0.016
0.059
Ex4DGS
31.34
0.016
0.050
31.01
0.016
0.051
FTGS
31.44
0.017
0.047
31.34
0.017
0.047
Ours (GraphiGS)
31.78
0.015
0.044
31.76
0.015
0.044
Ours (GraphiTS)
31.87
0.014
0.043
31.85
0.015
0.043
Table 5. Comparison with Random Faulty Cameras.
Method
Setting 1
Setting 2
PSNR↑
DSSIM↓
LPIPS↓
PSNR↑
DSSIM↓
LPIPS↓
4DGS-1
30.77
0.017
0.051
30.31
0.020
0.055
4DGS-2
30.56
0.018
0.061
29.78
0.021
0.065
Ex4DGS
30.52
0.019
0.053
30.03
0.020
0.056
FTGS
30.46
0.020
0.051
29.90
0.026
0.051
Ours (GraphiGS)
30.90
0.018
0.048
30.39
0.020
0.052
Ours (GraphiTS)
30.79
0.019
0.050
30.18
0.021
0.053
unsynchronized cameras. We simulate this scenario by setting cam-
eras at different FPS. We randomly select 10% and 50% of cameras
and reduce their FPS from 30 to 20, yielding 2 settings with two
levels of partial temporal sparsity. Table 4 shows the results. Overall,
we find Unsynchronized Cameras is an easier setting for all methods
compared with Sparse Views and Sparse Frames. This is expected as
all methods learn the dynamics of the components which does not
require observations to be available for arbitrary time. Unsynchro-
nized Cameras could be a bigger issue if cameras are not dense in
space where every frame from every camera is crucial in capturing
the motions. But in N3DV, all the cameras are in front of the person.
Overall, GraphiXS gives the best results.
4.5
Comparison under Faulty Cameras
During data recording, any camera can malfunction which will cause
a combination of all the types of the data uncertainty before this
section. This data uncertainty spans across space and time. We design
two settings to simulate Faulty Camera. Both settings aggregate
the previous parameters, resulting in total space-time sparsities of
approximately 13% for Setting 1 and 37% for Setting 2. Table 5
shows the numerical results for both settings. One visual example is
shown in Fig. 1 Right and more are in Fig. 6.
4.6
GraphiXS as an Upgrade
GraphiXS is not only a specific method, but a framework that can be
used to ‘upgrade’ other methods. This involves turning their deter-
ministic models into probabilistic ones, using formulations similar
to Eq. (9), then modeling stochasticity. The former step varies de-
pending on the specific method, and the latter step is to add our
stochastic components such as 𝑃(𝛼| 𝜃,𝐶,𝑇) and 𝑃(Σ𝑡
𝑝). We directly
show results here and give the details in the SM. We use the Standard
Setting and the Faulty Camera setting as it includes all types of data
uncertainty. Among the latest methods, we choose FTGS [Wang et al.
2025b] and Table 6 shows the numerical results. One visual example
is shown in Fig. 7.
Table 6. Comparison in before and after upgrade for FTGS across
Standard and Faulty Camera settings.
Method
Standard
Faulty Cam 1
Faulty Cam 2
PSNR↑DSSIM↓LPIPS↓PSNR↑DSSIM↓LPIPS↓PSNR↑DSSIM↓LPIPS↓
FTGS
31.55
0.016
0.047
30.46
0.020
0.051
29.90
0.026
0.051
FTGS UG 31.61
0.016
0.044
30.80
0.019
0.050
30.20
0.021
0.054
4.7
Ablation Study
Since we decompose GraphiXS into distributions (Eq. (5)) and pro-
pose various parameterization for each distribution, we show their
respective effectiveness. The key distributions include the Higher
Order Dynamics (Eq. (12)) and the component confidence 𝑃(𝛼|
𝛿𝜃,𝐶,𝑇)(Eq. (10)). Therefore, we conduct an ablation study on GraphiGS
under the two settings of Faulty Cameras. We show the quantitative
results in Table 7 and qualitative results in Fig. 8.
In general, removing any component will deteriorate the results.
This is more obvious in Setting 2 which has a higher level of uncer-
tainty. ‘W/O Higher Order Dynamics’ only considers the position
and velocity of the component, which is a strategy for many existing
4DGS methods [Lee et al. 2024; Li et al. 2024; Wang et al. 2025b].
But the results clearly shows that considering higher order dynamics
will improve the results. Furthermore, ‘W/O 𝑃(𝛼| 𝜃,𝐶,𝑇)’ shows
that the component confidence part also improves the results. This
part is designed to regulate component locations and shapes to make
them visible to cameras in time. In other words, every component
should maximize its utility to the whole model.
Table 7. Quantitative results of our ablation study on GraphiGS.
Method
Faulty Cams
Setting 1
Setting 2
PSNR↑
DSSIM↓
LPIPS↓
PSNR↑
DSSIM↓
LPIPS↓
W/O Higher Order Dynamics
30.84
0.019
0.049
30.03
0.021
0.054
W/O 𝑃(𝛼| 𝜃,𝐶,𝑇)
30.74
0.019
0.051
30.04
0.022
0.054
Ours (GraphiGS)
30.90
0.018
0.048
30.39
0.020
0.052
5
Conclusion and Discussion
We have proposed the first probabilistic 4DGS framework, GraphiXS,
that holistically considers multiple types of data uncertainty. GraphiXS
is general in that it can be instantiated with different components or
used to upgrade existing deterministic methods. Through exhaustive
evaluation and comparison, we have demonstrated the effectiveness
of GraphiXS. A major limitation is GraphiXS assumes the compo-
nents are probabilistic distributions parameterized by a set of com-
mon parameters. This makes it unsuitable for upgrading methods
with other types of components such as geometric primitives. Also,
GraphiXS does not do full Bayesian inference, i.e. no posterior dis-
tribution learned for 𝜃, which would be ideal given the stochastic
nature of 4DGS models.
Acknowledgments
This work was supported in part by the Rabin Ezra Scholarship Trust
(Charity No. 1116049), awarded to Do˘ga Yılmaz.

<!-- page 8 -->
8
•
Yilmaz et al.
References
Tim Bailey and Hugh Durrant-Whyte. 2006. Simultaneous localization and mapping:
part II. IEEE robotics & automation magazine 13, 3 (2006), 108–117.
Minh-Quan Viet Bui, Jongmin Park, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong
Oh, and Munchurl Kim. 2025. MoBGS: Motion Deblurring Dynamic 3D Gaussian
Splatting for Blurry Monocular Video. arXiv preprint arXiv:2504.15122 (2025).
Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu.
2024. High-quality surface reconstruction using gaussian surfels. In ACM SIGGRAPH
2024 conference papers. 1–11.
Junli Deng, Ping Shi, Qipei Li, and Jinyang Guo. 2025. DynaSplat: Dynamic-Static
Gaussian Splatting with Hierarchical Motion Decomposition for Scene Reconstruc-
tion. arXiv preprint arXiv:2506.09836 (2025).
Hugh Durrant-Whyte and Tim Bailey. 2006. Simultaneous localization and mapping:
part I. IEEE robotics & automation magazine 13, 2 (2006), 99–110.
Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning
Xu, Zhilin Pei, Hengjie Li, et al. 2025. Flashgs: Efficient 3d gaussian splatting for
large-scale and high-resolution rendering. In Proceedings of the Computer Vision and
Pattern Recognition Conference. 26652–26662.
Qiankun Gao, Yanmin Wu, Chengxiang Wen, Jiarui Meng, Luyang Tang, Jie Chen,
Ronggang Wang, and Jian Zhang. 2024.
Relaygs: Reconstructing dynamic
scenes with large-scale and complex motions via relay gaussians. arXiv preprint
arXiv:2412.02493 (2024).
Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa Choudhuri, Terrence Chen,
and Ziyan Wu. 2025. 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting.
arXiv preprint arXiv:2503.07946 (2025).
Fengzhi Guo, Chih-Chuan Hsu, Sihao Ding, and Cheng Zhang. 2025. Uncertainty
Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction. arXiv
preprint arXiv:2510.12768 (2025).
Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl
Vondrick, Bernard Ghanem, and Andrea Vedaldi. 2024. Ges: Generalized exponential
splatting for efficient radiance field rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 19812–19822.
Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio
Giancola, Andrea Vedaldi, Bernard Ghanem, and Marc Van Droogenbroeck. 2025. 3D
convex splatting: Radiance field rendering with 3D smooth convexes. In Proceedings
of the Computer Vision and Pattern Recognition Conference. 21360–21369.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2d
gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH
2024 conference papers. 1–11.
Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho. 2024. RoDyGS: Ro-
bust Dynamic Gaussian Splatting for Casual Videos. arXiv preprint arXiv:2412.03077
(2024).
Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. 2023.
Consistent4d:
Consistent 360 {\deg} dynamic object generation from monocular video. arXiv
preprint arXiv:2311.02848 (2023).
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
4 (2023), 139–1.
Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre
Lanvin, and George Drettakis. 2024. A hierarchical 3d gaussian representation for
real-time rendering of very large datasets. ACM Transactions on Graphics (TOG) 43,
4 (2024), 1–15.
Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng,
Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 2024. 3d
gaussian splatting as markov chain monte carlo. Advances in Neural Information
Processing Systems 37 (2024), 80965–80986.
Hyunjin Kim, Haebeom Jung, and Jaesik Park. 2025. Metropolis-Hastings Sampling for
3D Gaussian Reconstruction. arXiv preprint arXiv:2506.12945 (2025).
Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae, and Hae-Gon Jeon. 2024. Fully
explicit dynamic gaussian splatting. Advances in Neural Information Processing
Systems 37 (2024), 5384–5409.
Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner,
Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard New-
combe, et al. 2022. Neural 3d video synthesis from multi-view video. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition. 5521–5531.
Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024. Spacetime gaussian feature splatting
for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 8508–8520.
Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun
Liu. 2025. Longsplat: Robust unposed 3d gaussian splatting for casual long videos.
In Proceedings of the IEEE/CVF International Conference on Computer Vision.
27412–27422.
Changkun Liu, Shuai Chen, Yash Bhalgat, Siyan Hu, Ming Cheng, Zirui Wang, Vic-
tor Adrian Prisacariu, and Tristan Braud. 2024. GS-CPR: Efficient camera pose
refinement via 3d gaussian splatting. arXiv preprint arXiv:2408.11085 (2024).
Rong Liu, Zhongpai Gao, Benjamin Planche, Meida Chen, Van Nguyen Nguyen, Meng
Zheng, Anwesa Choudhuri, Terrence Chen, Yue Wang, Andrew Feng, et al. 2025a.
Universal Beta Splatting. arXiv preprint arXiv:2510.03312 (2025).
Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. 2025b. Deformable
beta splatting. In Proceedings of the Special Interest Group on Computer Graphics
and Interactive Techniques Conference Conference Papers. 1–11.
Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Fran-
cisco Vicente Carrasco, and Fernando De La Torre. 2024. Taming 3dgs: High-quality
radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers.
1–11.
Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. 2024. Gaussian
splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 18039–18048.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
mamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields
for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim,
Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew
Fitzgibbon. 2011. Kinectfusion: Real-time dense surface mapping and tracking. In
2011 10th IEEE international symposium on mixed and augmented reality. Ieee,
127–136.
Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong
Oh, and Munchurl Kim. 2025. Splinegs: Robust motion-adaptive spline for real-time
dynamic 3d gaussians from monocular video. In Proceedings of the Computer Vision
and Pattern Recognition Conference. 26866–26875.
Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021.
D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 10318–10327.
Luca Savant, Diego Valsesia, and Enrico Magli. 2024. Modeling uncertainty for gaussian
splatting. arXiv preprint arXiv:2403.18476 (2024).
Johannes L. Schönberger et al. 2016. COLMAP: Structure-from-Motion and Multi-View
Stereo. https://colmap.github.io/. Accessed: 2026-01-11.
Steven M Seitz, Brian Curless, James Diebel, Daniel Scharstein, and Richard Szeliski.
2006. A comparison and evaluation of multi-view stereo reconstruction algorithms. In
2006 IEEE computer society conference on computer vision and pattern recognition
(CVPR’06), Vol. 1. IEEE, 519–528.
Shimon Ullman. 1979. The interpretation of structure from motion. Proceedings of the
Royal Society of London. Series B. Biological Sciences 203, 1153 (1979), 405–426.
Toon Van de Maele, Ozan Çatal, Alexander Tschantz, Christopher L Buckley, and
Tim Verbelen. 2024.
Variational Bayes Gaussian Splatting.
arXiv preprint
arXiv:2410.03592 (2024).
Rui Wang, Quentin Lohmeyer, Mirko Meboldt, and Siyu Tang. 2025a.
Degauss:
Dynamic-static decomposition with gaussian splatting for distractor-free 3d recon-
struction. In Proceedings of the IEEE/CVF International Conference on Computer
Vision. 6294–6303.
Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun
Bao, Sida Peng, and Xiaowei Zhou. 2025b. FreeTimeGS: Free Gaussian Primitives
at Anytime Anywhere for Dynamic Scene Reconstruction. In CVPR. https://zju3dv.
github.io/freetimegs
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu
Liu, Qi Tian, and Xinggang Wang. 2024. 4d gaussian splatting for real-time dynamic
scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 20310–20320.
Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang,
Dong Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable
and versatile 3d generation. In Proceedings of the Computer Vision and Pattern
Recognition Conference. 21469–21480.
Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. 2025. Compositional
generative model of unbounded 4D cities. arXiv preprint arXiv:2501.08983 (2025).
Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu
Jiang. 2024. Physgaussian: Physics-integrated 3d gaussians for generative dynam-
ics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 4389–4398.
Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng Feng, and Li Zhang. 2024.
Dg-slam: Robust dynamic gaussian splatting slam with hybrid pose optimization.
Advances in Neural Information Processing Systems 37 (2024), 51577–51596.
Qitong Yang, Mingtao Feng, Zijie Wu, Weisheng Dong, Fangfang Wu, Yaonan Wang,
and Ajmal Mian. 2025. Hierarchical Gaussian Mixture Model Splatting for Efficient
and Part Controllable 3D Generation. In Proceedings of the Computer Vision and
Pattern Recognition Conference. 11104–11114.
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin.
2024a. Deformable 3d gaussians for high-fidelity monocular dynamic scene re-
construction. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 20331–20341.
Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. 2024b. Real-time Photorealis-
tic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting. In

<!-- page 9 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty
•
9
International Conference on Learning Representations (ICLR).
Do˘ga Yılmaz and Furkan Kıraç. 2023. Illumination-guided inverse rendering benchmark:
Learning real objects with few cameras. Computers & Graphics 115 (2023), 107–121.
doi:10.1016/j.cag.2023.07.002
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018.
The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings
of the IEEE conference on computer vision and pattern recognition. 586–595.
Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan
Yang. 2024. Drivinggaussian: Composite gaussian splatting for surrounding dynamic
autonomous driving scenes. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 21634–21643.
Jialin Zhu, Jiangbei Yue, Feixiang He, and He Wang. 2025. 3D Student Splatting
and Scooping. In Proceedings of the Computer Vision and Pattern Recognition
Conference. 21045–21054.
Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang,
Tianzhu Zhang, and Yongdong Zhang. 2024. Motiongs: Exploring explicit motion
guidance for deformable 3d gaussian splatting. Advances in Neural Information
Processing Systems 37 (2024), 101790–101817.
Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. 2002. EWA
splatting. IEEE Transactions on Visualization and Computer Graphics 8, 3 (2002),
223–238.

<!-- page 10 -->
10
•
Yilmaz et al.
Ground
Truth
4DGS-1
[Yang et al. 2024b]
4DGS-2
[Wu et al. 2024]
Ex4DGS
[Lee et al. 2024]
FTGS
[Wang et al. 2025b]
GraphiGS
(Ours)
GraphiTS
(Ours)
Fig. 3. Comparison under standard setting. Per-region PSNR scores are given at the top right of each image. Enlarged regions contain complex
and fast motions e.g. the tongs motion. GraphiXS reconstruction is more clear and richer in details than other methods.
Ground
Truth
4DGS-1
[Yang et al. 2024b]
4DGS-2
[Wu et al. 2024]
Ex4DGS
[Lee et al. 2024]
FTGS
[Wang et al. 2025b]
GraphiGS
(Ours)
GraphiTS
(Ours)
10%
30%
50%
Fig. 4. Comparison under 10%, 30%, and 50% spatial sparsity. Per-region PSNR scores are given at the top right of each crop. GraphiXS is
affected the least when the percentage of missing cameras increases.
Ground
Truth
4DGS-1
[Yang et al. 2024b]
4DGS-2
[Wu et al. 2024]
Ex4DGS
[Lee et al. 2024]
FTGS
[Wang et al. 2025b]
GraphiGS
(Ours)
GraphiTS
(Ours)
20 FPS
10 FPS
Fig. 5. Comparison under 20 FPS and 10 FPS temporal sparsity. Per-region PSNR scores are given at the top right of each crop. GraphiXS is
affected the least when the training camera FPS drops.

<!-- page 11 -->
Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty
•
11
Ground
Truth
4DGS-1
[Yang et al. 2024b]
4DGS-2
[Wu et al. 2024]
Ex4DGS
[Lee et al. 2024]
FTGS
[Wang et al. 2025b]
GraphiGS
(Ours)
GraphiTS
(Ours)
Faulty Cam 1
Faulty Cam 2
Fig. 6. Comparison under faulty camera 1 and faulty camera 2 settings. Per-region PSNR scores are given at the top right of each crop. GraphiXS
is affected the least when spatio-temporal sparsity is increased.
Ground
Truth
W/O Upgrade
W Upgrade
W/O Upgrade
W Upgrade
Faulty Cam 1
Faulty Cam 2
Fig. 7. Comparison of FTGS [Wang et al. 2025b] with and without upgrading under faulty camera 1 and 2 settings. Per-region PSNR scores are
given at the top right of each crop. Upgrading FTGS using GraphiXS improves visual quality under various levels of spatio-temporal uncertainty.
Ground
Truth
W/O Higher Order Dynamics
W/O 𝑃(𝛼| 𝜃,𝐶,𝑇)
GraphiTS (Ours)
Faulty Cam 1
Faulty Cam 2
Fig. 8. Visual results of our ablation study. Per-region PSNR scores are given at the top right of each crop.
