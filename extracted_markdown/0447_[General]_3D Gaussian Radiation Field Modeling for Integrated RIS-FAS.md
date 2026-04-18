<!-- page 1 -->
1
3D Gaussian Radiation Field Modeling for
Integrated RIS-FAS Systems: Analysis and
Optimization
Kaining Wang, Student Member, IEEE, Bo Yang, Senior Member, IEEE, Yusheng Lei, Zhiwen Yu, Senior
Member, IEEE, Xuelin Cao, Senior Member, IEEE, Liang Wang, Member, IEEE, Bin Guo, Senior Member, IEEE,
George C. Alexandropoulos, Senior Member, IEEE, M´erouane Debbah, Fellow, IEEE, and Zhu Han, Fellow, IEEE
Abstract—The integration of reconfigurable intelligent surfaces
(RIS) and fluid antenna systems (FAS) has attracted considerable
attention due to its tremendous potential in enhancing wireless
communication performance. However, under fast-fading channel
conditions, rapidly and effectively performing joint optimization
of the antenna positions in an FAS system and the RIS phase con-
figuration remains a critical challenge. Traditional optimization
methods typically rely on complex iterative computations, thus
making it challenging to obtain optimal solutions in real time
within dynamic channel environments. To address this issue, this
paper introduces a field information-driven optimization method
based on three-dimensional Gaussian radiation-field modeling
for real-time optimization of integrated FAS–RIS systems. In
the proposed approach, obstacles are treated as virtual trans-
mitters and, by separately learning the amplitude and phase
variations, the model can quickly generate high-precision channel
information based on the transmitter’s position. This design
eliminates the need for extensive pilot overhead and cumbersome
computations. On this framework, an alternating optimization
scheme is presented to jointly optimize the FAS position and the
RIS phase configuration. Simulation results demonstrate that the
proposed method significantly outperforms existing approaches
in terms of spectrum prediction accuracy, convergence speed,
and minimum achievable rate, validating its effectiveness and
practicality in fast-fading scenarios.
Index Terms—Fluid antenna system, reconfigurable intelligent
surface, 3D Gaussian Radiation Field, alternating optimization.
I. INTRODUCTION
W
ITH the rapid evolution of large-scale intelligent
wireless networks, the sixth-generation (6G) mobile
communications are envisioned to realize the ultimate goal
K. Wang, Y. Lei, B. Yang, L. Wang, and B. Guo are with the School of
Computer Science, Northwestern Polytechnical University, Xi’an, Shaanxi,
710129, China (email: wangkaining@mail.nwpu.edu.cn, yang bo, liangwang,
guob@nwpu.edu.cn).
Z. Yu is with the School of Computer Science, Northwestern Polytechnical
University, Xi’an, Shaanxi, 710129, China, and Harbin Engineering Univer-
sity, Harbin, Heilongjiang, 150001, China (email: zhiwenyu@nwpu.edu.cn).
X. Cao is with the School of Cyber Engineering, Xidian University, Xi’an,
Shaanxi, 710071, China (email: caoxuelin@xidian.edu.cn).
G. C. Alexandropoulos is with the Department of Informatics and Telecom-
munications, National and Kapodistrian University of Athens, 16122 Athens,
Greece (email: alexandg@di.uoa.gr).
M. Debbah is with KU 6G Research Center, Department of Computer and
Information Engineering, Khalifa University, Abu Dhabi 127788, UAE (email:
merouane.debbah@ku.ac.ae).
Z. Han is with the Department of Electrical and Computer Engineering
at the University of Houston, Houston, TX 77004 USA, and also with the
Department of Computer Science and Engineering, Kyung Hee University,
Seoul, South Korea, 446-701(email: hanzhu22@gmail.com).
of intelligent connectivity and programmable radio environ-
ments. Unlike conventional systems that passively adapt to
wireless propagation, 6G networks emphasize cognition and
reconfigurability, enabling communication nodes not only to
perceive but also to proactively reshape their surroundings for
dynamically optimal transmission. Among various enabling
technologies, the reconfigurable intelligent surface (RIS) has
emerged as a promising paradigm to achieve environment-
controllable communications [1]. By reprogramming the phase
shifts of its passive reflecting elements, RIS can redirect
incident electromagnetic (EM) waves toward desired direc-
tions without requiring additional radio-frequency (RF) chains,
thereby reconstructing the propagation environment in a cost-
and energy-efficient manner [2]–[4].
However, the performance of RIS-assisted systems in dy-
namic scenarios is fundamentally limited by channel non-
stationarity and pilot overhead. Owing to the passive and large-
scale nature of RIS, spectral leakage, phase coupling, and re-
flection misalignment in multi-user or multi-path environments
can significantly degrade system throughput [5], [6]. Existing
RIS optimization schemes rely heavily on iterative channel es-
timation and matrix-based optimization, whose computational
complexity grows exponentially with the number of reflecting
elements, thus making real-time configuration infeasible under
fast time-varying channels [7]–[9].
A. Motivation
To overcome the limitations of passive reflection, the fluid
antenna system (FAS) has been proposed as a flexible antenna
architecture capable of physically moving its antenna within a
small spatial region [10]–[12]. By continuously selecting the
most favorable reception position, FAS exploits small-scale
spatial diversity to enhance the desired signal and suppress
interference, all without additional bandwidth or power con-
sumption. Motivated by the complementary strengths of RIS
and FAS, their integration establishes a new communication
paradigm that enables joint adaptability across transmission,
propagation, and reception. In this framework, the RIS controls
large-scale wavefront shaping, whereas the FAS refines the re-
ceived signal through spatial sampling and position optimiza-
tion [13]–[16]. This hierarchical coordination enables dynamic
trade-offs between spectral efficiency, coverage uniformity,
and interference suppression.
arXiv:2511.01373v1  [cs.NI]  3 Nov 2025

<!-- page 2 -->
2
Nevertheless, the FAS–RIS paradigm still faces several
challenges. In particular, the positional mobility of the FAS
introduces a continuous spatial dimension to the channel
state information (CSI), making accurate characterization
of the radiation field across space increasingly demanding.
Consequently, conventional pilot-based estimation may lead
to excessive training overhead, especially when fine spatial
granularity is required. Moreover, the joint optimization of
RIS phase shifts and FAS positions involves multi-scale non-
convex coupling between reflection control and spatial field
distribution, which further complicates system design. While
analytical and ray-tracing models provide valuable insights
into the propagation characteristics, they are often limited in
capturing the fine spatial correlations inherent in FAS-assisted
systems. Meanwhile, emerging neural implicit representations,
such as NeRF [17] or volumetric field models, offer high
expressive power but usually at the expense of computational
efficiency and physical interpretability [18].
To address these limitations, this paper proposes a data-
driven optimization framework based on the three-dimensional
(3D) Gaussian radiation field (3DGRF) model. Unlike implicit
neural fields that rely on dense sampling and heavy training,
the 3DGRF represents the electromagnetic energy distribution
using explicit Gaussian primitives. Each primitive encodes
local amplitude and phase variations in a differentiable, phys-
ically consistent form, thereby enabling the efficient recon-
struction of the continuous radiation field from limited obser-
vations. This explicit, interpretable representation enables an-
alytical differentiation for gradient-based optimization of both
the RIS phase configuration and FAS position control. Conse-
quently, the proposed framework transforms traditional CSI-
dependent adaptation into radiation-field-driven sensing and
control, achieving low-complexity, real-time, and cognition-
driven optimization for next-generation FAS–RIS systems.
B. Related Work
1) Optimization of RIS–FAS Systems: RIS can program-
matically manipulate wireless propagation via passive phase
control, while FAS enables dynamic antenna-position switch-
ing within a confined region to exploit spatial diversity and
suppress interference. Their integration introduces a new class
of environment-adaptive architectures for 6G networks.
Early studies primarily focused on theoretical modeling of
RIS–FAS systems. For example, Lai et al. [19] developed an
analytical framework for outage probability under a block-
correlation model, while Yao et al. [20] introduced a block-
diagonal matrix approximation to simplify FAS port corre-
lation and derive outage bounds. After this, Yao et al. [21]
proposed joint optimization under both full and partial CSI
conditions, showing that CSI-free methods can outperform
conventional approaches in high-mobility environments. These
works provide valuable theoretical foundations but rely on
simplified or statistical channel assumptions, limiting their
ability to capture spatial non-stationarity and continuous field
variation.
From the optimization perspective, Tang et al. [22] min-
imized transmit power in multi-user RIS–FAS systems via
alternating optimization of beamforming, RIS phase shifts, and
FAS positions, assuming ideal CSI and static geometry. Yao et
al. [23] compared the performance of FAS and adaptive RIS
(ARIS), showing their complementarity in different channel
conditions. In terms of signal design, Zhu et al. [24] pro-
posed an FAS-based index modulation scheme for RIS-assisted
mmWave systems, while Ghadi et al. [25] analyzed FAS-
based RSMA and STAR-RIS systems under phase mismatch,
revealing that FAS can mitigate phase and hardware impair-
ments. Similarly, [26] demonstrated the potential of RIS–FAS
integration for physical-layer security and blockage resilience
by leveraging additional spatial degrees of freedom. Despite
these advances, most studies remain confined to small-scale
or idealized channel models and lack scalable, data-driven
methods for spatially continuous field inference.
2) Wireless Channel Modeling: Accurate wireless channel
modeling lies at the core of modern communication system
design. Probabilistic models capture large-scale path loss and
small-scale fading but neglect spatial correlation and direc-
tional propagation characteristics. Deterministic methods, such
as ray tracing (RT) [27], offer physically accurate multipath
modeling but incur exponential computational costs as scene
complexity increases, limiting their applicability in dynamic
environments. With the emergence of machine learning and
computer vision, data-driven modeling has gained traction.
NeRF-based methods [17] and NeWRF [18] reconstruct 3D ra-
diation fields from sparse samples, while physically informed
models such as WinERT [28], WiseGRT [29], and FIRE [30]
combine physical priors with neural representations to im-
prove generalization. Other approaches, such as R2F2 [31]
and OptML [32], employ cross-frequency CSI prediction and
differentiable channel estimation to enhance scalability.
Nevertheless, these models exhibit an inherent trade-off be-
tween accuracy, interpretability, and computational efficiency.
The ray tracing and NeRF-based frameworks achieve high
fidelity but lack real-time capability, whereas lightweight neu-
ral or statistical models compromise on spatial realism. To
bridge this gap, this paper introduces a 3DGRF framework
tailored for RIS–FAS systems. By representing EM energy
distributions with Gaussian primitives, the proposed approach
unifies RIS reflection modulation and FAS spatial sampling
into a continuous, differentiable radiation field. This physi-
cally grounded formulation transforms channel modeling from
matrix-based estimation to field-driven optimization, achieving
real-time reconstruction, interpretability, and adaptability in
complex 6G wireless environments.
C. Contributions and Organization
We detail our contributions as follows:
• A novel channel modeling approach based on 3D Gaus-
sian (3DGS) power-spectrum reconstruction is proposed
and applied to maximize the minimum achievable rate.
Unlike traditional methods that rely on extensive pilot-
based estimation, the proposed framework learns the 3D
Gaussian power-spectrum distribution of the environment
to achieve compact, efficient channel reconstruction. This
approach enables the rapid generation of high-fidelity

<!-- page 3 -->
3
channel information with limited training overhead, pro-
viding a reliable foundation for maximizing the minimum
achievable rate. In this way, the dependence on pilot sig-
nals is significantly reduced, while the system robustness
and adaptability under fast-fading conditions are greatly
enhanced.
• A comprehensive analysis of pilot overhead and in-
terference management in the FAS–RIS architecture is
conducted for the first time. While RIS enhances signal
coverage, it inevitably introduces spectral leakage and
interference coupling. FAS, on the other hand, mitigates
such interference by dynamically adjusting the receiver
antenna position. This work quantitatively investigates
the pilot requirements and interference suppression capa-
bility of the FAS–RIS system under various propagation
conditions, revealing the underlying synergy that enables
joint performance improvement. The findings provide
new insights into the feasibility and potential of deploying
FAS–RIS in future large-scale communication systems.
• An efficient joint optimization algorithm is designed
to achieve collaborative configuration of RIS and FAS
using power-spectrum priors. The proposed algorithm
fully exploits the prior information derived from the
reconstructed 3DGS power spectrum, thereby avoiding
the computational burden of high-dimensional iterative
searches in conventional optimization. By incorporating
power-spectrum priors as constraints in the optimiza-
tion process, the proposed method achieves near-optimal
phase control and position selection with low compu-
tational complexity, ensuring real-time adaptability and
scalability under fast-fading conditions.
• Extensive simulations validate the effectiveness and su-
periority of the proposed framework. The results demon-
strate that the proposed 3DGS reconstruction model
significantly outperforms existing methods in channel
reconstruction accuracy while achieving higher spectral
efficiency, lower computational complexity, and reduced
pilot overhead. Moreover, joint optimization of FAS and
RIS effectively mitigates interference and enhances sys-
tem capacity and fairness in multi-user scenarios.
The rest of this paper is organized as follows. Section II
presents the proposed system model and problem formulation;
Section III introduces the 3DGRF Gaussian radiation-field
modeling; Section IV describes the Field-Driven Alternating
Optimization (FAO) algorithm for joint FAS-RIS configura-
tion; Section V reports the simulation results and performance
comparisons. Finally, Section VI concludes the paper.
Notations: Scalars are denoted by uppercase italics unless
specified otherwise; vectors and matrices are denoted by bold
italic lowercase and bold uppercase letters, respectively. The
sets of complex and real numbers are denoted by C and R,
respectively. The following notations are used throughout this
paper: C denotes the complex field; (·)H denotes the conjugate
transpose; | · |2 denotes the ℓ2-norm. For matrix X, X⊤
and XH denote transpose and Hermitian. ℜ{·}, ℑ{·} extract
real/imaginary parts. For vector x, ∥x∥2 is the ℓ2-norm and
∥X∥F is Frobenius norm.
Fig.
1:
The
considered
system
model
of
integrated
RIS–FAS–assisted uplink multi-user communications.
II.
SYSTEM MODEL AND PROBLEM FORMULATION
A. Signal Modeling
As illustrated in Fig. 1, we consider a multiuser FAS-RIS-
assisted uplink communication system, where a total of U
users transmit signals simultaneously toward the base station
(BS). Among them, user k denotes the desired user, while
other users j act as interference sources, where k ̸= j.
Each user is equipped with a fixed-position antenna (FPA),
whereas the BS has M two-dimensional FASs. We define
ξ = [ξ1, ξ2, . . . , ξM] to represent the antenna positions at
the BS side. Within each FAS, every antenna element can
instantaneously switch to any location within the region Sξ =
[0, W] × [0, W].
The received signal at the m-th BS antenna from user k can
be expressed as
ym =
√
P hrmΘhH
kr + σ2,
(1)
where P denotes the transmit power and σ2 represents the
additive white Gaussian noise (AWGN) power at the BS
receiver.
The RIS consists of N reflecting elements, capable of
flexibly controlling both the amplitude and phase of the
reflected signals to manipulate the propagation direction of
electromagnetic waves. The cascaded reflection coefficient
matrix is denoted by
Θ = diag(ϕ1, ϕ2, . . . , ϕN),
(2)
where ϕn = exp(jθn), and n ∈N = {1, . . . , N}, where
θn ∈[0, 2π].
B. Channel Modeling
Since the RIS is deployed close to the user side, the
RIS–UE link is dominated by line-of-sight (LoS) propaga-
tion. However, the channels between the fluid antennas (FAs)
and the surrounding nodes, namely the UE–FA and RIS–FA
links, exhibit spatially correlated characteristics rather than
independent random fading. This is because the FA can move
within a small local region, where the wireless channel varies
smoothly with spatial position. Modeling these channels as
spatially correlated functions of the FA position ensures that
optimizing the FA location yields physically consistent and
meaningful performance gains.

<!-- page 4 -->
4
Let the m-th FA be located at position ξm = [xm, ym]T .
The RIS–FA channel between the RIS and FA m is modeled
as
hrm =
p
αrmRrm grm,
(3)
where αrm denotes the large-scale path loss, Rrm represents
the spatial correlation matrix [33], [34] describing the corre-
lation of the RIS–FA link, and grm is a small-scale fading
vector whose phase varies smoothly with the FA position.
The spatial correlation matrix Rrm can be modeled as
[Rrm] = ρ|ξm−ξ′
m|
rm
,
0 ≤ρrm ≤1,
(4)
where ρrm characterizes the spatial correlation between ad-
jacent RIS–FA subchannels. A larger ρrm indicates stronger
correlation due to the limited movement range of the FA within
the local area.
Similarly, the UE–FA link is expressed as
hkm =
p
αkmRkmgkm,
(5)
where αkm is the path loss of the UE–FA link, Rkm is the
corresponding spatial correlation matrix, and gkm denotes the
normalized local scattering vector dependent on FA position.
In the above expression, √Rkm is the Hermitian square
root, which captures the spatial dependency among the UE–FA
subchannels. It can be modeled using an exponential correla-
tion structure as
[Rkm] = ρ|ξm−ξ′
m|
km
,
0 ≤ρkm ≤1,
(6)
where ρkm denotes the spatial correlation coefficient that
reflects the similarity of fading between neighboring FA po-
sitions. The term √Rkm ensures that the generated channel
vectors maintain the desired correlation pattern while preserv-
ing the overall power normalization.
Consequently, the composite effective channel observed by
the UE is given by
hk = hkm + hkrΘhrm,
(7)
where hkr denotes the LoS RIS–UE channel.
C. Problem Definition
Based on the above signal and channel models, the signal-
to-interference-plus-noise ratio (SINR) at BS antenna m can
be expressed as
Γ =

√
P hrmΘhH
kr

2
UP
j=1,j̸=k

√
P hrmΘhH
jr

2
+ σ2
.
(8)
Accordingly, the minimum achievable rate from multiple users
to the BS is defined as
R = log2(1 + Γ).
(9)
Therefore, the objective is to maximize the minimum
achievable rate by jointly optimizing the RIS reflection co-
efficient matrix Θ and the BS antenna positions ξ. The
optimization problem can be formulated as
max
ξ,Θ
R
s.t.
ξ ∈Sξ,
∥ξm −ξv∥2 ≥D,
m, v ∈M, m ̸= v,
Θ = diag{ϕ1, . . . , ϕN},
ϕn = exp(jθn),
θn ∈[0, 2π),
(10)
where D denotes the minimum inter-antenna spacing to avoid
mutual coupling. However, due to strong coupling among the
optimization variables in both the objective function and the
constraints, as well as the problem’s highly nonconvex nature,
solving it poses significant challenges.
III. 3DGS-BASED RADIATION FIELD
RECONSTRUCTION
A. Preliminaries
Accurate modeling and reconstruction of wireless channels
constitute the foundation for realizing environment-adaptive
communication systems. However, conventional channel mod-
eling approaches encounter significant limitations in scenarios
involving the integration of RIS and FAS. The main challenge
arises from the fact that RIS–FAS systems simultaneously in-
volve multi-scale spatial wavefront manipulation and receiver-
side position adaptation. Consequently, the channel character-
istics dynamically vary with spatial locations and reflection
states, making it difficult for static statistical or deterministic
models to provide effective characterization.
Traditional probabilistic models rely on empirical formulas
and statistical fitting, which can only capture average proper-
ties such as path loss and large-scale fading but fail to represent
the fine-grained electromagnetic field distribution in space. As
a result, these models exhibit considerable errors in multiuser
and multipath RIS scenarios. Deterministic models, such as
ray tracing, offer high physical interpretability but suffer from
exponential computational complexity as environmental com-
plexity increases. Moreover, they heavily depend on precise
geometric modeling and material parameters, thus making
real-time application in dynamic scenarios infeasible. In addi-
tion, deep-learning-based implicit channel modeling methods
(e.g., the NeRF family) can learn nonlinear mappings between
environmental geometry and signal propagation. Nevertheless,
their training and rendering overheads are prohibitively high,
preventing their practical use for real-time channel estimation
and configuration in FAS–RIS systems.
In this context, the introduction of 3DGS provides an effi-
cient and physically consistent paradigm for channel modeling
in RIS–FAS systems. Originating from radiance field modeling
in computer vision, 3DGS can reconstruct high-fidelity spatial
field distributions from a limited number of sampled points in
continuous 3D space. Compared with implicit representations
such as NeRF, 3DGS adopts explicit Gaussian primitives,
achieving orders-of-magnitude advantages in both training
and rendering efficiency. More importantly, it facilitates the

<!-- page 5 -->
5
Fig. 2: Overall framework of the proposed 3DGRF–based optimization. (1) The environment setup includes BS, RIS, and FAS.
(2) The scenario representation network encodes geometric and transmitter information into latent radiation-field features.
(3) The projection model represents the electromagnetic field using differentiable 3DGS primitives. (4) The electromagnetic
splatting process reconstructs the continuous radiation field and generates spatial power distributions. (5) The reconstructed
field produces spatial spectrums for subsequent system optimization. (6) The FAO iteratively updates FAS positions and RIS
phases based on 3DGRF, achieving field-based control.
integration of electromagnetic propagation physics, enabling
joint reconstruction of amplitude and phase information.
B. Continuous Radiation Field Description
Traditional channel models typically represent the wireless
channel as the superposition of a finite number of discrete
propagation paths, expressed as
h(ξ) =
L
X
ℓ=1
Aℓejζℓ,
(11)
where L denotes the number of propagation paths, Aℓand
ζℓrepresent the complex gain and wave vector of the ℓ-th
path, respectively. The implicit assumption is that L is limited
and that the received signal variations arise only from the
interference among these discrete components.
However, in practical scenarios—especially in environments
involving RIS and spatially correlated FAs—the number of
scattering and reflection paths can be extremely large, and their
spatial distributions are often continuous rather than discrete.
This observation aligns with the spatially correlated channel
model discussed in the previous subsection, in which the
received signal varies smoothly with the FA position.
To capture this physical continuity, the wireless channel can
be described as a continuous radiation field rather than a finite-
path summation, i.e.,
h(ξ) =
Z
Ω
A(ℓ)ej2πζξ dℓ,
(12)
where A(ξ) denotes the spatial field spectrum associated with
the spatial frequency component ℓ, and Ωrepresents the
angular domain. This formulation bridges the discrete mul-
tipath representation and the spatially correlated field model,
providing a unified and physically consistent framework for
describing channel variations in FAS–RIS systems.
This continuous field representation further lays the foun-
dation for the subsequent field-driven optimization of RIS and
FAS configurations.
To characterize the continuous distribution of energy and
phase evolution in 3D space, we define a complex-valued
radiation field function E(ξ), representing the complex field
strength at an arbitrary spatial observation position:
E(ξ) =
L
X
ℓ=1
Aℓejθℓe−j 2π
λ ∥ξ−qℓ∥,
(13)
where ξ denotes the spatial observation position (FA location),
ql denotes the position of the virtual emitter corresponding to
the l-th path, and Al and θl represent its amplitude attenuation
and phase shifts, respectively. This model no longer explicitly
traces individual paths but instead describes the electromag-
netic field distribution at any location through a continuous
complex field E(ξ). Such a representation transforms the
propagation environment from a “set of discrete paths” to a
“spatially continuous field,” a key step toward unified field-
based modeling.
C. 3D Gaussian Primitives
To approximate the continuous radiation field with a finite
set of parameters, we employ 3DGS Primitives to model the
field in a distributed manner. Originally developed for scene
rendering in computer graphics, Gaussian primitives efficiently
represent continuous spatial distributions of light energy with
a small number of parameters. When adapted to the wireless
communication domain, they can effectively describe both the
amplitude and phase distributions of electromagnetic energy
in 3D space. The 3DGS approach directly approximates the
radiation energy cloud using analytical Gaussian functions,
thereby avoiding volumetric integration and significantly re-
ducing rendering and training overhead. This explicit represen-

<!-- page 6 -->
6
tation naturally supports differentiable optimization, making it
highly suitable for real-time updates of dynamic parameters in
FAS-RIS systems.
The i-th Gaussian primitive, located at position qi, is
defined as:
Gi(ξ) = Aiejζi exp

−1
2(ξ −qi)TΣ−1
i (ξ −qi)

,
(14)
where Ai denotes the amplitude coefficient of the Gaussian
primitive, reflecting its reflection or radiation strength, and
ζi represents the associated phase factor. The covariance
matrix Σi determines the spatial spread and anisotropy of the
primitive [35].
The overall radiation field is obtained by superimposing all
NG Gaussian primitives, i.e.,
˜E(ξ) =
NG
X
i=1
Gi(ξ),
(15)
where NG denotes the total number of Gaussian primitives
used to approximate the field distribution. A larger NG allows
for a finer representation of complex radiation patterns, while a
smaller NG yields a more compact but coarser approximation.
This formulation mathematically corresponds to a kernel
expansion of the spatial energy density function and, phys-
ically, can be interpreted as the coherent superposition of
electromagnetic fields generated by multiple virtual emission
points.
D. Gaussian Mapping Modeling for RIS-FAS Systems
Since the reflection positions and phase shifts of each unit
are fixed after configuration, they can be directly mapped to
deterministic Gaussian primitive parameters. For the n-th RIS
element, the corresponding phase and spatial center parameters
are defined as:
ψn = θn + 2π
λ (dk,n + dξ,n) ,
µn = qn,
(16)
where dk,n and dξ,n denote the propagation distances from the
transmitter and receiver to the n-th RIS element, respectively,
rn represents the 3D spatial coordinates of the element, ψn
is the phase parameter of the n-th Gaussian primitive (deter-
mined by the RIS phase shift θn and propagation distances),
and µn is the spatial center of the primitive.
Accordingly, the RIS-reflected field can be expressed as a
coherent superposition of Gaussian primitives:
Er(ξ) =
N
X
n=1
Anejψn exp

−1
2(ξ −µn)TΣ−1
n (ξ −µn)

,
(17)
where An is the amplitude coefficient of the n-th RIS element
(reflecting its reflection strength), and Σn is the covariance
matrix determining the spatial spread of the primitive.
This formulation explicitly incorporates both the phase
parameter ψn and spatial position µn of each RIS element
into the Gaussian mapping process, thereby bridging the phys-
ical RIS configuration and the analytical 3D Gaussian field
representation. When the RIS configuration is updated, only
the phase shift θn (which determines ψn) and amplitude An
need to be adjusted to rapidly obtain the new field distribution
without re-solving the entire channel.
The FAS receiver antennas can move within a constrained
spatial region Sξ. The received signal of the FAS is denoted
as:
ym =
√
P ˜E(ξm) + σ2,
ξm ∈Sξ,
(18)
where ξm denotes the instantaneous spatial position of the m-
th FAS antenna within the allowed region.
E. Scenario Representation Network
Geometric parameters alone are insufficient to characterize
the complex environmental effects on electromagnetic prop-
agation. Therefore, we introduce a Scenario Representation
Network (SRN) to learn the signal-propagation behavior, fol-
lowing the deepSDF structure [36].
The SRN takes as input the transmitter position ptx and an
environmental point-cloud coordinate q, and outputs a pair of
complex attenuation parameters (µ(q), δ(q)), formulated as:
F : (q, ptx) →(µ(q), δ(q)).
(19)
For each Gaussian point, the complex coefficient is ex-
pressed as:
C(q) = µ(q)ejθ(q).
(20)
The SRN consists of two multilayer perceptrons (MLPs).
The first MLP extracts spatial geometric features and learns
the attenuation distribution associated with the environmental
position, while the second MLP fuses these features with the
transmitter position to predict the signal’s amplitude and phase
responses.
This structure ensures the adaptability of Gaussian param-
eters to environmental geometry, enabling rapid generation
of new Gaussian parameters under varying RIS reflection
configurations or obstacle layouts.
The network is trained by minimizing the structural-
similarity-aware reconstruction loss between the reconstructed
and the ground-truth fields. Let Egt(ξ) denote the measured
ground-truth radiation field at spatial observation position ξ,
and ˜E(ξ) represent the field reconstructed by the SRN. We
have the loss function as
L=(1−η)∥Egt(ξ)−˜E(ξ)∥2
2+η
 1−SSIM(Egt(ξ), ˜E(ξ))

, (21)
where η is a weighting factor, and the SSIM is the Structural
Similarity Index Measure function.
Proof. See Appendix A.
After training, the SRN can generate the corresponding
Gaussian radiation field for any given RIS configuration and
FAS position, enabling fast, physically consistent channel
reconstruction.

<!-- page 7 -->
7
F. Projection Model and Electromagnetic Splatting
The projection model maps the virtual TXs q, represented
by 3D Gaussians, onto the FA’s perception plane, ξ. To obtain
the angular-domain response of the FAS antenna array, the
3DGRF must be projected onto the array’s perceptual plane.
Since the receiving coverage of the array corresponds to a
hemispherical region, we employ a Mercator projection to
map spatial coordinates (xq, yq, zq) into angular coordinates
(Ωlon, Ωlat):





Ωlon = arctan 2(yq, xq),
Ωlat = arcsin
 
z1
p
x2
1 + y2
1 + z2
1
!
,
(22)
where Ωlon and Ωlat represent the longitude and latitude of
the angular domain, respectively.
The angular space is then discretized to obtain the spatial
power spectrum matrix:
P(Ωlon, Ωlat) =
 ˜E(ξ(Ωlon, Ωlat))
2.
(23)
Furthermore, at the implementation level, an efficient ac-
cumulation is achieved through the Electromagnetic Splatting
mechanism. Physically, electromagnetic splatting characterizes
the energy attenuation and phase accumulation of multipath
signals at different propagation depths, serving as the radio-
frequency counterpart of optical ”light blending”.
After each Gaussian primitive is projected onto the 2D
angular plane, its contribution is accumulated sequentially
according to depth order. Let Cq(Ωq) denote the complex
signal of the q-th Gaussian primitive at angular position Ωq.
The total received signal is given by:
Rk =
NG
X
i=1


i−1
Y
j=1
µ(qj)ejδ(qj)

Cq(Ωq),
(24)
and the corresponding power spectrum is:
I(Ωk) = |Rk|2.
(25)
This parallel splatting process is implemented on GPUs,
allowing millisecond-level generation of spatial power spectra
and enabling real-time visualization of the FAS–RIS channel.
By integrating the Gaussian mapping of RIS reflection
elements and the spatial sampling of the FAS receiver, the
complete system radiation field can be formulated as:
Ek(ξ) =
J
X
i=1
Aiejθi exp
 −1
2(ξ −qi)TΣ−1
n (ξ −qi)

, (26)
where J is the number of 3D Gaussians near a given pixel.
Proof. See Appendix B.
The received signal at the m-th FAS antenna position ξm ∈
Sξ is expressed as:
ym =
√
P Ek(ξm) + σ2.
(27)
IV. PROPOSED FAS-RIS DESIGN
In this section, to effectively tackle the joint optimization
problem formulated in (10), we decompose it into two interde-
pendent sub-problems: the continuous optimization of the FAS
positions ξ and the discrete optimization of the RIS reflection
phase matrix Θ. Leveraging the differentiable 3DGRF model,
we propose a FAO framework that iteratively updates these
two sets of variables to achieve a locally optimal solution.
A. Problem Reformulation
Building upon the 3DGRF formulation established in Sec-
tion III, we now reformulate the joint optimization problem
of the RIS-FAS system in a field-driven manner. As defined
in (18) and (27), the received signal at the m-th FA can be
expressed as
ym =
√
P Ek(ξm) +
X
j̸=k
√
P Ej(ξm) + σ2, ξm ∈St,
(28)
where Ek(ξm) denotes the desired-user field distribution de-
fined in (26), Ej(ξm) represents the interference field con-
tributed by user j.
Based on the previously established 3DGRF and the re-
ceived signal, the instantaneous angular power spectrum of
the system is defined as:
Φ(ξ, Θ) = 1
M
M
X
m=1
|Ek(ξm)|2 ,
(29)
which directly characterizes the spatial distribution of the
radiation field’s energy.
The interference power spectrum is given by
Φj(ξ, Θ) = 1
M
M
X
m=1
X
j̸=k
|Ej(ξm)|2 .
(30)
The overall received SINR can then be expressed as
Γ(ξ, Θ) =
P Φ(ξ, Θ)
PΦj(ξ, Θ) + σ2 .
(31)
Accordingly, the optimization objective R can be rewritten
as (9). The joint optimization problem is then formulated as:
max
s,Θ
log2
 
1 +
P
1
M
PM
m=1
Θkm
2
P 1
M
PM
m=1
P
j̸=k
Θjm
2 + σ2
!
s.t.
ξm ∈St, ∥ξm −ξv∥2 ≥D (m ̸= v),
θn ∈{0, 2π
Lc , . . . , 2π(1 −
1
Lc )},
(32)
where the inner summation term is simplified as
Θjm =
J
X
i=1
A(j)
i ejθi exp
 −1
2(ξm −qi)TΣ−1
i (ξm −qi)

where A(j)
i
denotes the amplitude coefficient of the i-th
Gaussians for FA m’s signal path.
Here, Lc represents the phase quantization level. This
problem is a typical mixed non-convex optimization problem,
where ξ is a continuous variable and Θ is discrete. Moreover,

<!-- page 8 -->
8
the strong coupling between these variables in the objective
function makes direct optimization intractable. To address
this issue, a FAO method is proposed. The core idea is
to iteratively update the FAS positions and RIS phases by
leveraging the power-spectrum gradients reconstructed from
the 3DGRF, rather than relying on explicit channel estimation
matrices.
B. FAS Position Optimization
When the RIS reflection matrix Θ is fixed, the subproblem
of optimizing the FAS positions can be formulated as:
max
ξ
Φ(ξ; Θ)
s.t. ξm ∈Sξ, ∥ξm −ξv∥2 ≥D.
(33)
Since the objective function Φ(ξ; Θ) is non-convex with
respect to ξ, we adopt the successive convex approximation
(SCA) technique to iteratively linearize the objective around
the current position ξ
(v) [37]. Specifically, the first-order
Taylor expansion is employed to construct a locally convex
surrogate function:
eΦ(ξ) = Φ(ξ
(v)) +
M
X
m=1
∇ξmΦ(ξ
(v))⊤(ξm −ξ(v)
m ),
(34)
where the gradient ∇ξmΦ is given by
∇ξmΦ =
N
X
n=1
ℜ

E∗
k(ξm)∂Ek(ξm)
∂ξm

.
(35)
The convexified subproblem in the (q + 1)-th iteration can
thus be written as:
max
ξ
eΦ(ξ)
s.t. ξm ∈Sξ, ∥ξm −ξv∥2 ≥D.
(36)
This convex optimization can be efficiently solved using
standard solvers. After convergence, the antenna positions are
updated as
ξ(v+1)
m
= ξ(v)
m + ϑξ(ξ∗
m −ξ(v)
m ),
(37)
where ϑξ is the adaptive step-size ensuring monotonic im-
provement of the objective. Through successive convex re-
finements, the FAS positions gradually converge to a locally
optimal configuration that maximizes the radiation-field power
spectrum.
The overall FAS position optimization is summarized in
Algorithm 1.
C. RIS Phase Configuration Optimization
When the FAS positions s are fixed, the optimization of the
RIS phase shifts can be expressed as:
max
Θ
Φ(ξ; Θ)
s.t. θn ∈C,
(38)
where C denotes the discrete codebook set [38] of phase
quantization levels.
To efficiently explore the non-convex discrete search space,
we employ a genetic algorithm (GA) [39]. Each individual in
the GA population represents a candidate RIS phase vector
Algorithm 1 FAS Position Optimization under Fixed RIS
Require: Fixed RIS Θ, initial FAS positions ξ
(0), step size
ϑξ, threshold ϵξ, max iterations Qξ
Ensure: Optimized FAS positions ξ
∗
for v = 0 to Qξ do
Compute gradient ∇ξmΦ as in (35)
Update ξm using (37) and project onto Sξ
Compute Φ(v+1) using (29)
if |Φ(v+1) −Φ(v)| < ϵξ then
break
end if
end for
return ξ
∗= ξ
(v+1)
θ = [θ1, . . . , θN], and its fitness function is defined as the
corresponding radiation power:
F(θ) = Φ(ξ; Θ(θ)).
(39)
For each RIS element n, the marginal power contribution
of a candidate phase θ is defined as:
∆Pn(θ) = F([θ1, . . . , θn−1, θ, θn+1, . . . , θN]) −F(θ). (40)
During the optimization process, the GA evolves the pop-
ulation through three key operations: selection, crossover,
and mutation. In each generation, individuals with higher
fitness values are preferentially selected to preserve promising
phase configurations. The crossover operation then combines
partial phase vectors from selected pairs of individuals to
generate new offspring, enabling information exchange among
high-quality solutions. To prevent premature convergence and
maintain diversity, the mutation step randomly perturbs several
phase entries with a small probability. Over QGA generations
of evolution, the population gradually converges to an optimal
or near-optimal RIS phase configuration.
After QGA generations, the best individual θ∗is adopted as
the optimized RIS phase configuration:
Θ∗= diag{ejθ∗
1, ejθ∗
2, . . . , ejθ∗
N }.
(41)
The GA-based discrete optimization effectively avoids local
minima and enables near-global search without explicit gradi-
ent computation.
The overall Discrete Optimization of RIS Phases is summa-
rized in Algorithm 2.
D. Field-Driven Alternating Optimization Framework
The overall optimization framework alternates between the
SCA-based continuous optimization of FAS positions and the
GA-based discrete optimization of RIS phases, as summarized
in Algorithm 3.
Each iteration guarantees a non-decreasing radiation-field
power, and since the overall objective is upper-bounded, the
algorithm converges to a locally optimal field configuration.
The optimization process operates directly on the radiation-
field representation, enabling adaptive control of both FAS
positions and RIS phases without relying on explicit channel
estimation or convex relaxations.

<!-- page 9 -->
9
Algorithm 2 RIS Phase Optimization under Fixed FAS
Require: Fixed FAS s, initial RIS Θ(0), codebook C, thresh-
old ϵr, max iterations Qr
Ensure: Optimized RIS phases Θ∗
for v = 0 to Qr do
for n = 1 to N do
Compute ∆Pn(θn) as in (40)
Select θ(v+1)
n
= arg maxθ∈C ∆Pn(θ)
end for
Form updated RIS Θ(v+1) = diag(ejθ(v+1)
1
, . . . , ejθ(v+1)
N
)
Compute Φ(v+1) using (29)
if |Φ(v+1) −Φ(v)| < ϵr then
break
end if
end for
return Θ∗= Θ(v+1)
Algorithm 3 Field-Driven Optimization using 3DGRF
Require: Initial FAS ξ
(0), RIS Θ(0), max iterations Qmax,
threshold ϵ
Ensure: Optimized ξ
∗, Θ∗
for v = 0 to Qmax do
FAS Update: Fix Θ(v), update ξ
(v+1) using (35)–(37)
RIS Update: Fix ξ
(v+1), update Θ(v+1) using (40) and
(41)
Compute Φ(v+1) using (29)
if |Φ(v+1) −Φ(v)| < ϵ then
break
end if
end for
return ξ
∗= ξ
(v+1), Θ∗= Θ(v+1)
In terms of computational complexity, the SCA-based FAS
position update primarily involves gradient-based matrix-
vector operations that scale linearly with the number of FAS
elements M. Meanwhile, the GA-based RIS phase update
requires evaluating the fitness of N reflecting elements with
Lc bits. Therefore, the overall per-iteration complexity of the
proposed framework can be expressed as
O(M + N |Lc|),
(42)
which reflects the linear scalability of the proposed field-
driven optimization with respect to the system dimensions.
This ensures that the framework can efficiently adapt to large-
scale RIS-assisted deployments in real time.
E. Complexity Analysis
The computational complexity of the proposed alternating
optimization framework mainly arises from two stages: the
SCA-based FAS position update and the GA-based RIS phase
optimization. In each outer iteration, denoted by Q, these
two modules are executed sequentially to refine the field
configuration.
In the FAS optimization stage, the dominant operations
come from evaluating the radiation-field gradient and solv-
ing the convexified subproblem in (36). For each antenna
element, the gradient ∇ξmΦ in (35) requires accumulating
the contributions of all N RIS reflecting elements, which
results in a computational cost of O(MN). The subsequent
projection and position update in (37) involve only element-
wise operations and therefore add negligible overhead. Hence,
the overall complexity of the FAS update per iteration can be
approximated as
CFAS ≈O(MN).
(43)
In the RIS optimization stage, the genetic algorithm eval-
uates the fitness function defined in (39) for a population
of Gpop individuals over Giter generations. Each evaluation
involves computing the total radiation power contributed by
N reflecting elements, leading to a per-generation cost of
O(GpopN). Therefore, the total computational complexity of
the GA-based phase optimization can be expressed as
CRIS ≈O(GpopGiterN).
(44)
Combining both stages, the overall complexity of one outer
iteration of the proposed framework is given by
Citer = CFAS + CRIS ≈O(MN + GpopGiterN).
(45)
After Q alternating iterations, the total computational com-
plexity can thus be summarized as
O(Q(MN + GpopGiterN)) .
(46)
Since both the gradient computations and fitness evaluations
are highly parallelizable, the proposed field-driven optimiza-
tion can be efficiently implemented on GPUs. In practice,
the number of antennas M and reflecting elements N are
moderate, while Gpop and Giter are typically small to ensure
real-time convergence. As a result, the proposed framework
enables fast reconfiguration, well-suited to dynamic 6G com-
munication environments.
V. NUMERICAL RESULTS AND DISCUSSION
A. Implementation for 3DGRF Reconstruction
3DGRF experiments are conducted on a workstation
equipped with an NVIDIA RTX 4090 GPU with CUDA
kernels, an Intel(R) Silver(R) 4410Y CPU, and 128 GB RAM.
The 3DGRF reconstruction and optimization are implemented
using PyTorch 2.3.1 and CUDA 12.2. All GPU operations,
including Gaussian rendering and field-projection operations,
are parallelized to achieve real-time training and inference.
We use an open-source dataset provided in [17] to eval-
uate the 3DGRF framework. The physical environment for
radiation-field reconstruction is shown in Fig. 3. A 3D LiDAR
sensor is employed to capture the surrounding geometry,
including walls, tables, and reflectors. The 3D LiDAR point
clouds of the environment are shown in Fig. 3(a). The receiver
(RX) is equipped with a 4 × 4 antenna array, while the trans-
mitter (TX) continuously sends messages. The RX position
is fixed, and the TX position is systematically varied within
the laboratory. Each data sample comprises a TX coordinate

<!-- page 10 -->
10
Fig. 3: (a)Laboratory environment and LiDAR point-cloud representation for radiation-field reconstruction, where the base
station (BS) and multiple users Uk, U1, Uj are marked. 3D LiDAR point clouds of the laboratory environment were used
to initialize the Gaussian primitives for the proposed 3DGS-based field model. The overall experimental scene, where the
transmitter (TX) can be placed at arbitrary positions within the environment, and the receiver (RX) equipped with a 4 × 4
antenna array is fixed at the corner of the room.
and the corresponding measured spatial spectrum at the RX.
The dataset includes 6,000 samples, with 80% for training
and 20% for testing. The obtained 3D point cloud provides
a dense geometric representation of the laboratory environ-
ment, containing approximately 1.5×106 points. These points
are preprocessed by downsampling and normal estimation to
initialize the spatial distribution of Gaussian primitives. Each
LiDAR point corresponds to an initial Gaussian center µq,
whose covariance Σq is determined by local point density and
surface curvature, thereby encoding geometric smoothness into
the initial radiation-field model. The position and orientation of
the Gaussian primitives are updated adaptively during network
optimization.
1) Scene Representation Network: The scene representation
network (SRN) consists of two MLPs. The first MLP encodes
the spatial geometry of each Gaussian primitive and predicts
amplitude attenuation µ(q), while the second predicts phase
offset δ(q) conditioned on the transmitter position. Each
MLP contains four hidden layers with 256 neurons and uses
LeakyReLU activation. The network is trained with Adam
optimizer using an initial learning rate of 5 × 10−4, decayed
exponentially by 0.95 every 20 epochs.
The loss function jointly measures amplitude fidelity and
structural similarity between the reconstructed and ground-
truth fields, as shown in (21).
2) Radiation Field Reconstruction Baselines: To evaluate
the fidelity of the proposed 3DGRF reconstruction, we com-
pare it against three representative baselines:
• Generative Adversarial Network (GAN) [40]: A data-
driven benchmark that directly learns the nonlinear map-
ping from transmitter or RIS–FAS configurations to the
corresponding power or radiation field distributions. The
generator captures global spatial correlations through
TABLE I. Simulation Parameters
Parameter
Symbol
Value
Number of RIS elements
N
64
Phase quantization level
Lc
4 (2-bit)
Number of FAS antennas
M
16
Minimum spacing
W
λ
Transmit power
P
10 dBm
Noise power
σ2
−90 dBm
Convergence threshold
ε
10−4
Learning rate
lr
5 × 10−4
adversarial training against a discriminator.
• Variational Autoencoder (VAE) [41]: A probabilistic
generative model that embeds the field distribution into a
latent Gaussian manifold. By sampling from the learned
latent space, the VAE reconstructs continuous radiation
maps with smooth energy variations.
• Probabilistic
Channel
Model
(PCM): A physics-
inspired baseline that characterizes the wireless prop-
agation using stochastic statistical parameters such as
path loss, Rician fading, and spatial correlation matrices.
Optimization is performed in the channel domain by
iteratively updating RIS phases and FAS positions based
on estimated CSI.
B. System Parameters for Communications
The main simulation parameters for the communication
environment are summarized in Table I.
To comprehensively evaluate the proposed FAO framework,
we compare it with two categories of reference schemes: (i)
optimization methods that evaluate the effectiveness of the
proposed algorithm, and (ii) system baselines that reveal the
importance of FAS and RIS for achieving the rate maximum.

<!-- page 11 -->
11
1) Optimization Methods:
• GD [42]: A classical continuous-phase optimization
method that updates each RIS reflection coefficient
through projected gradient descent under the unit-
modulus constraint. After convergence, the optimized
continuous phases are uniformly quantized to a 2-bit
codebook [43]. This approach reflects conventional differ-
entiable optimization performed in the channel domain.
• ADMM [44]: A physics-consistent convex-relaxation
framework that decomposes the coupled FAS–RIS joint
optimization into several tractable subproblems with sep-
arable constraints. Each subproblem can be solved effi-
ciently with guaranteed convergence to a locally optimal
stationary point.
Both serve as algorithmic baselines that approximate the
physical-space optimization process but remain channel-
dependent, providing insight into the advantages of direct
field-domain optimization achieved by 3DGRF-driven FAO.
2) System Baselines:
• w/o RIS: The RIS module is deactivated, and only the
direct BS–user transmission is considered. This baseline
represents a conventional FAS-assisted system without
any reflective enhancement and serves to quantify the net
gain introduced by RIS deployment.
• Random:The RIS is activated with random discrete phase
assignments selected from a 2-bit quantization. This
case evaluates the effect of random scattering without
optimization.
• FPA: The fluid antenna system operates with fixed an-
tenna positions, while the RIS is fully optimized. This
baseline removes the mobility degree of freedom of FAS,
revealing the pure benefit of spatial adaptability brought
by antenna movement.
These baselines jointly quantify the impact of each functional
component—RIS reconfiguration and FAS movement—on
overall system performance, enabling a clear demonstration of
the effectiveness of the proposed 3DGRF-driven optimization
framework.
C. Radiation Field Modeling Performance
Fig. 3(b) compares the angular-domain power spectra re-
constructed by different radiation-field modeling methods,
including the proposed 3DGS, GAN, and VAE. The ground-
truth spectrum is obtained from real-world ray-tracing mea-
surements calibrated by site-specific channel sounding, serving
as a reference for field reconstruction. As observed, the
3DGS model accurately reproduces both the global radiation
pattern and the fine interference fringes of the measured field,
exhibiting almost perfect alignment with the ground truth. In
contrast, the GAN and VAE baselines exhibit clear spatial
blurring and phase inconsistencies, particularly around high-
energy lobes and null regions. These results verify that the
explicit Gaussian-primitive representation in 3DGS effectively
preserves sub-wavelength spatial energy variations and phase
continuity, leading to physically faithful field modeling. Such
high-fidelity reconstruction provides a reliable foundation for
subsequent field-driven RIS–FAS optimization, ensuring that
(a)
(b)
Fig. 4: (a) Synthesized spatial power spectrum of the RIS–FAS
system before optimization, where the integration of RIS
reflection and FAS reception forms a preliminary directional
pattern. (b) Spectrum after field-driven optimization, showing
stronger beam focusing and reduced sidelobes, which demon-
strates the effectiveness of the proposed joint configuration of
RIS phase and FAS position.
the learned radiation field can faithfully guide configuration
decisions in real propagation environments.
D. Visualization of Radiation Field Spectrum
In Fig. 4(a), the spatial spectrum corresponds to the initial
configuration after integrating the FAS and RIS modules. The
radiation energy becomes directionally enhanced compared
with the case in Fig. 3(b), indicating that the joint reflec-
tion–reception structure already forms a preliminary beam
pattern toward the dominant propagation direction. After per-
forming the proposed field-driven alternating optimization,
Fig. 4(b) shows that the main lobe becomes significantly
narrower and the sidelobes are largely suppressed, demon-
strating a well-focused beamforming effect. This observation
confirms that the optimized RIS phase and FAS position
jointly steer the electromagnetic field to the desired spatial
region, achieving stronger energy concentration and more
efficient spatial utilization. The result verifies the physical
interpretability of the 3DGRF model, which directly translates
field-domain optimization into observable beam enhancement.
E. Transmission Rate Comparison
Fig. 5 illustrates the evolution of the minimum achiev-
able rate versus iteration number for the proposed field-
driven optimization and the conventional optimization under
identical system settings. As the iteration count increases,
the achievable rates of both methods gradually stabilize,
indicating that each algorithm reaches convergence. Notably,
the proposed approach converges within fewer iterations and
attains a markedly higher steady-state rate, demonstrating both
faster convergence and superior performance. This perfor-
mance gain originates from the 3DGRF. By leveraging the
learned power spectrum, the optimizer can rapidly align the
transceiver configuration with the actual energy distribution
in the environment, thereby improving spectral efficiency and
robustness in dynamic or fast-fading conditions. These results
verify that the explicit Gaussian field modeling bridges the

<!-- page 12 -->
12
Fig. 5: Minimum achievable rate R comparison between the
proposed field-driven optimization and the traditional opti-
mization.
Fig. 6: Average delay versus different channel modeling meth-
ods.
gap between physical-space representation and communication
optimization. It achieves fast, stable convergence without
requiring channel estimation, while maintaining low compu-
tational complexity and pilot overhead.
F. Average Delay Results
Fig. 6 compares the average processing latency of four
optimization frameworks that employ different channel mod-
eling paradigms: the proposed 3DGS model, VAE, GAN,
and PCM–based optimization. All methods share the same
inference and optimization pipeline, while differing only in
how the underlying propagation environment is represented
and updated. As shown, the proposed 3DGS-based method
achieves the lowest latency, with an average computation time
of only 5 × 10−3 s, which is about 10× faster than the
probabilistic model–based traditional optimization (0.2 s) and
significantly more efficient than the data-driven baselines VAE
(0.1 s) and GAN (0.05 s).
This demonstrates that the 3DGS framework not only con-
verges faster but also achieves real-time adaptability with min-
imal computational overhead. The gain arises from the 3DGS
model’s radiation-field reconstruction mechanism, which di-
rectly infers the receiver-side radio-frequency field from the
transmitter geometry, enabling continuous field representation
without repeated channel estimation or iterative matrix in-
version. In contrast, the VAE and GAN baselines can also
infer the radiation field but rely on deep encoder–decoder
Fig. 7: Transmit power versus minimum achievable rate R.
architectures with large parameter spaces, resulting in longer
inference times and higher GPU memory usage. The proba-
bilistic channel model further incurs heavy iterative computa-
tion due to stochastic channel sampling and expectation-based
optimization. Results show that the proposed 3DGS model
provides a lightweight and physically interpretable alternative
that supports low-latency, real-time RIS–FAS optimization.
G. Achievable Sum-Rate Results
Fig. 7 illustrates the achievable rate performance versus
transmit power P ∈[−10, 20]dBm under identical RIS–FAS
configurations. As expected, all curves exhibit a monotonic
increase with transmit power, confirming the theoretical SNR
dependence of achievable rate. The proposed FAO consistently
achieves the highest rate across the entire power range. This
superior performance stems from the 3DGRF modeling, which
provides a differentiable, physically grounded description of
both the desired signal focusing and the reflected interference
field. By learning the continuous radiation distribution, the
FAO jointly optimizes FAS positions and RIS phases to con-
centrate energy toward the desired user while simultaneously
suppressing undesired reflections from interfering users. This
field-domain coordination enables coherent power aggregation
and adaptive interference control, thereby maximizing the
minimum achievable rate in multiuser conditions. The GD and
ADMM schemes follow similar upward trends but reach lower
saturation levels. Their channel-domain formulations rely on
discrete CSI and iterative convex updates, which approximate
the alignment of reflection phases and cannot fully mitigate
inter-user interference in the continuous radiation field. The
w/o RIS baseline lacks reflective diversity, resulting in poor
energy utilization. The Random configuration suffers from
uncontrolled scattering and constructive–destructive phase ran-
domness, yielding severe interference fluctuations. The FPA
case fixes antenna positions, thereby losing the spatial adapt-
ability that enables FAS to exploit local radiation peaks, re-
sulting in a moderate but limited improvement. The consistent
superiority of the FAO curve across all power levels proves
that field-aware optimization provides better energy focusing
and robustness than channel-driven schemes, reinforcing the
efficiency of 3DGRF-based field modeling.

<!-- page 13 -->
13
Fig. 8: The number of RIS reflecting elements versus minimum
achievable rate R.
Fig. 8 shows the achievable rate versus the number of RIS
elements N. depicts the variation of the minimum achievable
rate with respect to the number of RIS elements N, ranging
from 16 to 256. A notable observation is that the proposed
field-driven alternating optimization (FAO) exhibits a steep
acceleration in achievable rate for N > 64. This sharp rise
is because a larger RIS aperture enhances interference sup-
pression by enabling finer control over reflected sidelobes, and
simultaneously amplifies coherent signal aggregation, since the
reflected wavefront can be more precisely aligned toward the
desired user. In contrast, the w/o RIS baseline remains nearly
flat because the system lacks any reflective enhancement—the
received power depends solely on the direct path, and increas-
ing N provides no benefit. The FPA curve starts below the w/o
RIS case, but later surpasses it as N grows. This is because,
without RIS assistance, the FPA cannot avoid interference nor
locate the strongest radiation zones; however, when the RIS
aperture becomes sufficiently large, its enhanced reflection
gain compensates for the lack of antenna mobility. A larger
RIS not only strengthens the effective signal power through
coherent combining but also enables fine-grained control of in-
terference fields. Meanwhile, the FA remains indispensable. Its
spatial adaptability complements RIS phase reconfiguration,
jointly achieving robust interference suppression and efficient
energy focusing in large-aperture deployments.
Fig. 9 illustrates the minimum achievable rate as a function
of the RIS horizontal position along the x-axis, while the
receiver is fixed at the origin and the transmit power is kept
constant. The RIS is gradually moved away from the user
toward the BS, covering a range of x ∈[1, 8]m. The achievable
rate first decreases and then increases as the RIS moves along
the x-axis. This phenomenon arises from the multiplicative
path-loss effect between the user–RIS and RIS–BS links. FAO
consistently outperforms the baselines across all positions. Its
advantage lies in 3DGRF modeling, which captures spatial
field variations caused by RIS relocation and dynamically re-
optimizes both the RIS phase and the FAS position to maintain
coherent energy focusing and interference suppression. In
contrast, the w/o RIS case remains nearly constant because it
lacks reflective enhancement. The Random baseline exhibits
noticeable fluctuations because its discrete random phase con-
Fig. 9: RIS x-axis versus Achievable rate versus minimum
achievable rate R.
Fig. 10: The normalized FAs’ movable range W/λ versus
minimum achievable rate R.
figuration leads to inconsistent constructive and destructive
interference, underscoring the importance of precise RIS phase
control. The FPA baseline remains below the proposed method
because fixed antennas cannot spatially adapt to field changes.
Placing the RIS near either endpoint enhances both the
coherent reflection gain and the controllability of interference.
Fig. 10 illustrates the achievable rate versus the normalized
FAS movement range W/λ. As W/λ increases, all methods
exhibit steady performance improvement, indicating that a
larger FAS movement region provides more spatial degrees
of freedom. With a wider exploration space, the antenna
can select positions with stronger radiation intensity and
reduced interference, thereby improving the received SINR.
The proposed FAO achieves the highest growth rate and the
steepest slope across all ranges. Its 3DGRF model accurately
captures the fine-scale variation in the electromagnetic field,
enabling the antenna to move adaptively to the most favor-
able positions. By contrast, GD and ADMM exhibit slower
improvement because they rely on discrete CSI estimation
and iterative convex approximations. As the movement re-
gion enlarges, its optimization space grows exponentially.
The w/o RIS baseline improves only marginally since FAS
mobility alone cannot fully compensate for the absence of
reflective gain. Expanding the FAS movable range increases

<!-- page 14 -->
14
the system’s spatial adaptability, enabling joint exploitation
of field peaks and interference nulls. However, as the search
space grows, optimization efficiency becomes the performance
bottleneck. Therefore, the combination of advanced field-
driven algorithms and flexible FAS hardware is essential for
achieving high-rate, interference-resilient communication in
next-generation 6G networks.
VI. CONCLUSIONS
This paper has presented a novel field-driven optimiza-
tion framework for the FAS-RIS system based on 3DGRF
modeling. By explicitly representing the electromagnetic en-
ergy distribution using differentiable Gaussian primitives, the
proposed method transforms the conventional channel-driven
paradigm into a radiation-field-based optimization process.
This unified formulation enables joint optimization of FAS
positions and RIS phase shifts with low pilot overhead and
computational complexity, achieving real-time adaptability in
fast-fading and spatially non-stationary environments. Com-
prehensive simulations validated that the proposed framework
achieves superior spectral efficiency, convergence speed, and
latency performance compared with traditional channel esti-
mation–based or statistical modeling approaches. The results
confirm that 3DGRF modeling not only enhances the physical
interpretability of the optimization process but also bridges the
gap between wireless field reconstruction and communication
configuration, offering a scalable and energy-efficient solution
for next-generation 6G systems.
APPENDIX A
Let ξm denote the spatial observation position of the m-
th receiver antenna. The ground-truth electromagnetic field
Egt(ξm) is obtained from the theoretical model as
Egt(ξm) =
√
P hkm +
N
X
n=1
√
P hknejθnhnm,
(47)
where hkm, hkn, and hnm denote the direct and RIS-reflected
channel coefficients.
The SRN aims to reconstruct ˜E(ξ) from the sampled field,
and is optimized by minimizing the hybrid loss function
L = (1 −η)∥Egt(ξ) −˜E(ξ)∥2
2 + η
 1 −SSIM(Egt(ξ), ˜E(ξ))

,
(48)
where η ∈[0, 1] is the balancing coefficient. The SSIM is
defined as
SSIM(x, y) =
(2µxµy + C1)(2σxy + C2)
(µ2x + µ2y + C1)(σ2x + σ2y + C2),
(49)
with µx, µy, σ2
x, σ2
y, and σxy denoting mean, variance, and
covariance, respectively.
APPENDIX B
A continuous spatial field E(s) can be approximated by M
Gaussian kernels:
E(ξ) ≈
M
X
i=1
Gi(ξ).
(50)
For an isotropic emitter centered at µi with variance σ2
i , we
have
|Gi(ξ)| = Ai exp

−∥ξ −µi∥2
2σ2
i

.
(51)
For anisotropic spatial distributions, the covariance matrix Σi
replaces σ2
i , yielding
|Gi(ξ)| = Ai exp

−1
2(ξ −qi)TΣ−1
i (ξ −qi)

.
(52)
Extending to complex-valued representation, each Gaussian
primitive is defined as
Gi(ξ) = Aiejζi exp

−1
2(ξ −qi)TΣ−1
i (ξ −qi)

,
(53)
where Ai and ψi denote the amplitude and phase, respectively.
Hence, the Gaussian primitive in (53) corresponds to the
analytical form used in the main text.
REFERENCES
[1] C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and
C. Yuen, “Reconfigurable intelligent surfaces for energy efficiency in
wireless communication,” IEEE Transactions on Wireless Communica-
tions, vol. 18, no. 8, pp. 4157–4170, 2019.
[2] E. Basar, G. C. Alexandropoulos, Y. Liu, Q. Wu, S. Jin, C. Yuen, O. A.
Dobre, and R. Schober, “Reconfigurable intelligent surfaces for 6G:
Emerging hardware architectures, applications, and open challenges,”
IEEE Veh. Technol. Mag., vol. 19, no. 3, pp. 27–47, 2024.
[3] K. Wang, B. Yang, Z. Yu, X. Cao, M. Debbah, and C. Yuen, “Filtering
reconfigurable intelligent computational surface for RF spectrum purifi-
cation,” IEEE Network, vol. 39, no. 1, pp. 63–70, 2025.
[4] Q. Wu, B. Zheng, C. You, L. Zhu, K. Shen, X. Shao, W. Mei, B. Di,
H. Zhang, E. Basar, L. Song, M. Di Renzo, Z.-Q. Luo, and R. Zhang,
“Intelligent surfaces empowered wireless network: Recent advances and
the road to 6G,” Proc. IEEE, vol. 112, no. 7, pp. 724–763, 2024.
[5] G. C. Alexandropoulos, M. Crozzoli, D.-T. Phan-Huy, K. D. Katsanos,
H. Wymeersch, P. Popovski, P. Ratajczak, Y. B´en´edic, M.-H. Hamon,
S. H. Gonzalez, R. D’Errico, and E. C. Strinati, “Smart wireless environ-
ments enabled by riss: Deployment scenarios and two key challenges,”
in 2022 Joint European Conference on Networks and Communications
& 6G Summit (EuCNC/6G Summit), 2022, pp. 1–6.
[6] G. Stamatelis, P. Gavriilidis, A. Fakhreddine, and G. C. Alexandropou-
los, “On the detection of non-cooperative RISs: Scan B-testing via
deep support vector data description,” in ICC 2025-IEEE International
Conference on Communications.
IEEE, 2025, pp. 6844–6849.
[7] K. Stylianopoulos, P. Gavriilidis, and G. C. Alexandropoulos, “Asymp-
totically optimal closed-form phase configuration of 1-bit RISs via
sign alignment,” in 2024 IEEE 25th International Workshop on Signal
Processing Advances in Wireless Communications (SPAWC), 2024, pp.
746–750.
[8] A. L. Moustakas, G. C. Alexandropoulos, and M. Debbah, “Reconfig-
urable intelligent surfaces and capacity optimization: A large system
analysis,” IEEE Transactions on Wireless Communications, vol. 22,
no. 12, pp. 8736–8750, 2023.
[9] A. L. Moustakas and G. C. Alexandropoulos, “MIMO MAC empowered
by reconfigurable intelligent surfaces: Capacity region and large system
analysis,” IEEE Transactions on Wireless Communications, vol. 23,
no. 12, pp. 19 245–19 258, 2024.
[10] F. Rostami Ghadi, K.-K. Wong, W. K. New, H. Xu, R. Murch, and
Y. Zhang, “On performance of RIS-aided fluid antenna systems,” IEEE
Wireless Communications Letters, vol. 13, no. 8, pp. 2175–2179, 2024.
[11] B. Tang, H. Xu, K.-K. Wong, L. You, J. Tang, Y. Zhang, and H. Shin,
“Power minimization of multiuser FAS-RIS downlink system,” IEEE
Transactions on Vehicular Technology, pp. 1–6, 2025.
[12] J. Yao, T. Wu, L. Zhou, M. Jin, C. Huang, and C. Yuen, “FAS vs. ARIS:
Which is more important for FAS-ARIS communication systems?” IEEE
Transactions on Wireless Communications, pp. 1–1, 2025.
[13] R. Xu, Z. Yang, Z. Zhang, M. Shikh-Bahaei, K. Huang, and D. Niyato,
“Energy efficient fluid antenna relay (FAR)-assisted wireless communi-
cations,” IEEE Journal on Selected Areas in Communications, pp. 1–1,
2025.

<!-- page 15 -->
15
[14] H. Xiao, X. Hu, K.-K. Wong, H. Hong, G. C. Alexandropoulos,
and C.-B. Chae, “Fluid reconfigurable intelligent surfaces: Joint on-off
selection and beamforming with discrete phase shifts,” IEEE Wireless
Communications Letters, vol. 14, no. 10, pp. 3124–3128, 2025.
[15] A. Salem, K.-K. Wong, G. Alexandropoulos, C.-B. Chae, and R. Murch,
“A first look at the performance enhancement potential of fluid recon-
figurable intelligent surface,” arXiv preprint arXiv:2502.17116, 2025.
[16] J. Yao, X. Lai, K. Zhi, T. Wu, M. Jin, C. Pan, M. Elkashlan, C. Yuen,
and K.-K. Wong, “A framework of FAS-RIS systems: Performance
analysis and throughput optimization,” IEEE Transactions on Wireless
Communications, pp. 1–1, 2025.
[17] X. Zhao, Z. An, Q. Pan, and L. Yang, “NeRF2: Neural radio-frequency
radiance fields,” in Proceedings of the 29th Annual International
Conference on Mobile Computing and Networking, ser. ACM MobiCom
’23.
New York, NY, USA: Association for Computing Machinery,
2023. [Online]. Available: https://doi.org/10.1145/3570361.3592527
[18] H. Lu, C. Vattheuer, B. Mirzasoleiman, and O. Abari, “NeWRF: a
deep learning framework for wireless radiation field reconstruction and
channel prediction,” in Proceedings of the 41st International Conference
on Machine Learning, ser. ICML’24.
JMLR.org, 2024.
[19] X. Lai, J. Yao, K. Zhi, T. Wu, D. Morales-Jimenez, and K.-K. Wong,
“FAS-RIS: A block-correlation model analysis,” IEEE Transactions on
Vehicular Technology, 2024.
[20] J. Yao, J. Zheng, T. Wu, M. Jin, C. Yuen, K.-K. Wong, and F. Adachi,
“FAS-RIS communication: Model, analysis, and optimization,” IEEE
Transactions on Vehicular Technology, vol. 74, no. 6, pp. 9938–9943,
2025.
[21] J. Yao, X. Lai, K. Zhi, T. Wu, M. Jin, C. Pan, M. Elkashlan, C. Yuen,
and K.-K. Wong, “A framework of FAS-RIS systems: Performance
analysis and throughput optimization,” IEEE Transactions on Wireless
Communications, pp. 1–1, 2025.
[22] B. Tang, H. Xu, K.-K. Wong, L. You, J. Tang, Y. Zhang, and H. Shin,
“Power minimization of multiuser FAS-RIS downlink system,” IEEE
Transactions on Vehicular Technology, pp. 1–6, 2025.
[23] J. Yao, T. Wu, L. Zhou, M. Jin, C. Huang, and C. Yuen, “FAS vs. ARIS:
Which is more important for FAS-ARIS communication systems?” IEEE
Transactions on Wireless Communications, pp. 1–1, 2025.
[24] J. Zhu, Q. Luo, G. Chen, P. Xiao, Y. Xiao, and K.-K. Wong, “Fluid
antenna empowered index modulation for RIS-aided mmwave transmis-
sions,” IEEE Transactions on Wireless Communications, vol. 24, no. 2,
pp. 1635–1647, 2025.
[25] F. R. Ghadi, K.-K. Wong, M. Kaveh, F. J. Lopez-Martinez, Y. Liu, C.-B.
Chae, and R. Murch, “Phase-mismatched STAR-RIS with FAS-assisted
RSMA users,” arXiv preprint arXiv:2503.08986, 2025.
[26] F. R. Ghadi, K.-K. Wong, M. Kaveh, F. J. Lopez-Martinez, W. K. New,
and H. Xu, “Secrecy performance analysis of RIS-aided fluid antenna
systems,” in 2025 IEEE Wireless Communications and Networking
Conference (WCNC).
IEEE, 2025, pp. 1–6.
[27] H. Choi, J. Oh, J. Chung, G. C. Alexandropoulos, and J. Choi, “Withray:
A versatile ray-tracing simulator for smart wireless environments,” IEEE
Access, vol. 11, pp. 56 822–56 845, 2023.
[28] T. Orekondy, P. Kumar, S. Kadambi, H. Ye, J. Soriaga, and A. Behboodi,
“WiNeRT: Towards neural ray tracing for wireless channel modelling
and differentiable simulations,” in The Eleventh International Confer-
ence on Learning Representations, 2023.
[29] L. Zhang, H. Sun, J. Sun, and R. Q. Hu, “WiSegRT: Dataset for
site-specific indoor radio propagation modeling with 3D segmentation
and
differentiable
ray-tracing,”
2023.
[Online].
Available:
https:
//arxiv.org/abs/2312.11245
[30] Z. Liu, G. Singh, C. Xu, and D. Vasisht, “FIRE: enabling reciprocity for
FDD MIMO systems,” in Proceedings of the 27th Annual International
Conference on Mobile Computing and Networking, ser. MobiCom ’21.
New York, NY, USA: Association for Computing Machinery, 2021, p.
628–641. [Online]. Available: https://doi.org/10.1145/3447993.3483275
[31] D. Vasisht, S. Kumar, H. Rahul, and D. Katabi, “Eliminating channel
feedback in next-generation cellular networks,” in Proceedings of the
2016 ACM SIGCOMM Conference, ser. SIGCOMM ’16.
New York,
NY, USA: Association for Computing Machinery, 2016, p. 398–411.
[Online]. Available: https://doi.org/10.1145/2934872.2934895
[32] A. Bakshi, Y. Mao, K. Srinivasan, and S. Parthasarathy, “Fast and
efficient cross band channel prediction using machine learning,” in
The 25th Annual International Conference on Mobile Computing
and
Networking,
ser.
MobiCom
’19.
New
York,
NY,
USA:
Association for Computing Machinery, 2019. [Online]. Available:
https://doi.org/10.1145/3300061.3345438
[33] G. C. Alexandropoulos, N. C. Sagias, F. I. Lazarakis, and K. Berberidis,
“New results for the multivariate nakagami-m fading model with
arbitrary correlation matrix and applications,” IEEE Transactions on
Wireless Communications, vol. 8, no. 1, pp. 245–255, 2009.
[34] M. Matthaiou, G. C. Alexandropoulos, H. Q. Ngo, and E. G. Larsson,
“Analytic framework for the effective rate of miso fading channels,”
IEEE Transactions on Communications, vol. 60, no. 6, pp. 1741–1751,
2012.
[35] C. Wen, J. Tong, Y. Hu, Z. Lin, and J. Zhang, “WRF-GS: Wireless
radiation field reconstruction with 3D gaussian splatting,” in IEEE
INFOCOM 2025 - IEEE Conference on Computer Communications,
2025, pp. 1–10.
[36] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove,
“DeepSDF: Learning continuous signed distance functions for shape
representation,” in 2019 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2019, pp. 165–174.
[37] M. Razaviyayn, “Successive convex approximation: Analysis and appli-
cations,” Ph.D. dissertation, University of Minnesota, 2014.
[38] J. An, C. Xu, Q. Wu, D. W. K. Ng, M. Di Renzo, C. Yuen, and L. Hanzo,
“Codebook-based solutions for reconfigurable intelligent surfaces and
their open challenges,” IEEE Wireless Communications, vol. 31, no. 2,
pp. 134–141, 2022.
[39] Z. Peng, T. Li, C. Pan, H. Ren, W. Xu, and M. D. Renzo, “Analysis
and optimization for RIS-aided multi-pair communications relying on
statistical CSI,” IEEE Transactions on Vehicular Technology, vol. 70,
no. 4, pp. 3897–3901, 2021.
[40] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,”
Advances in neural information processing systems, vol. 27, 2014.
[41] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv
preprint arXiv:1312.6114, 2013.
[42] F. Zhu, X. Wang, C. Huang, Z. Yang, X. Chen, A. Al Hammadi,
Z. Zhang, C. Yuen, and M. Debbah, “Robust beamforming for RIS-
aided communications: Gradient-based manifold meta learning,” IEEE
Transactions on Wireless Communications, vol. 23, no. 11, pp. 15 945–
15 956, 2024.
[43] S. Zhang, H. Sun, R. Yu, H. Cui, J. Ren, F. Gao, S. Jin, H. Xie,
and H. Wang, “Two-bit RIS-aided communications at 3.5 GHz: Some
insights from the measurement results under multiple practical scenes,”
arXiv preprint arXiv:2305.11614, 2023.
[44] Z. Li, W. Chen, H. Qin, Q. Wu, X. Zhu, Z. Zhang, and J. Li, “Toward
TMA-based transmissive RIS transceiver enabled downlink communica-
tion networks: A consensus-ADMM approach,” IEEE Transactions on
Communications, vol. 73, no. 4, pp. 2832–2846, 2025.
