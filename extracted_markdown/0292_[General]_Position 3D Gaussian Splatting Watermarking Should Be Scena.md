<!-- page 1 -->
Position: 3D Gaussian Splatting Watermarking
Should Be Scenario-Driven and Threat-Model Explicit
Yangfan Deng 1 Anirudh Nakra 1 Min Wu 1
Abstract
3D content acquisition and creation are expanding
rapidly in the new era of machine learning and
AI. 3D Gaussian Splatting (3DGS) has become
a promising high-fidelity and real-time represen-
tation for 3D content. Similar to the initial wave
of digital audio-visual content at the turn of the
millennium, the demand for intellectual property
protection is also increasing, since explicit and
editable 3D parameterization makes unauthorized
use and dissemination easier. In this position pa-
per, we argue that effective progress in watermark-
ing 3D assets requires articulated security objec-
tives and realistic threat models, incorporating
the lessons learned from digital audio-visual asset
protection over the past decades. To address this
gap in security specification and evaluation, we
advocate a scenario-driven formulation, in which
adversarial capabilities are formalized through a
security model. Based on this formulation, we
construct a reference framework that organizes
existing methods and clarifies how specific design
choices map to corresponding adversarial assump-
tions. Within this framework, we also examine a
legacy spread-spectrum embedding scheme, char-
acterizing its advantages and limitations and high-
lighting the important trade-offs it entails. Over-
all, this work aims to foster effective intellectual
property protection for 3D assets.
1. Introduction
Recent advances in machine learning and AI are acceler-
ating the full lifecycle of 3D content, from capture and
reconstruction to creation and distribution. Consumer-grade
acquisition pipelines, neural rendering, and generative mod-
els are steadily reducing the cost of producing high-quality
1Department of Electrical and Computer Engineering, Uni-
versity of Maryland, College Park, MD, United States. Corre-
spondence to: Yangfan Deng <yfandeng@umd.edu>, Min Wu
<minwu@umd.edu>.
Preprint. February 4, 2026.
3D content. In parallel, real-time viewers and online market-
places make it increasingly convenient to package, transfer,
and repurpose 3D assets for downstream uses such as aug-
mented and virtual reality (AR/VR) (Dengel et al., 2022).
Consequently, 3D assets are emerging as a central modal-
ity in the digital media economy, where they are routinely
reproduced, edited, and redistributed at scale.
This trend has been observed repeatedly in digital media
evolution over the past several decades. When high-quality
digital audio, image, and video became easy to copy and dis-
tribute in the 1990s, intellectual property protection emerged
as a practical necessity, and watermarking became one of the
core technical tools being considered to support copyright
management and tracing under redistribution (Cox et al.,
2008; Stamm et al., 2013). Today, 3D assets face a similar
trade-off. Modern 3D representations are editable, which
improves usability but also makes misuse and unauthorized
redistribution easier. Among all 3D representations, 3D
Gaussian Splatting (3DGS) has quickly become a promi-
nent representation for high-fidelity novel-view synthesis
due to its explicit parameterization and real-time rendering
quality (Kerbl et al., 2023). Traditional 3D representations,
such as mesh-based pipelines, often require heavy recon-
struction and texturing. Point clouds may struggle to capture
complex appearance details (Wegen et al., 2024). And neu-
ral radiance field (NeRF) methods may struggle to represent
complex lighting (Mildenhall et al., 2021). In comparison,
3DGS offers an attractive balance between visual quality,
editability, and practical deployability. As a result, 3DGS
models are increasingly packaged and shared as transfer-
able assets in practical workflows (Guo et al., 2025; Peng
et al., 2024). Throughout this paper, we use 3DGS as a
concrete embodiment of this broader shift toward learnable
and redistributable 3D media assets.
As 3D assets become widely shared and repurposed in real-
world workflows fostered by advances in AI and machine
learning, protecting them against unauthorized redistribu-
tion and misuse becomes increasingly important. Water-
marking 3D assets is associated with security assumptions
and attack surfaces that are often more intricate than those
in traditional audio-visual media. In particular, protection
spans two coupled domains, the model domain of the 3D
1
arXiv:2602.02602v1  [cs.CR]  1 Feb 2026

<!-- page 2 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
representation and the rendering domain. The latter con-
sists of 2D renditions from potentially many viewpoints,
and an adversary may access either the 3D model or the
rendered outputs. In addition, 3DGS models are frequently
reparametrized and reordered during distribution and pro-
cessing, which can undermine index-anchored protection
mechanisms (Bagdasarian et al., 2025). Geometry and ap-
pearance are also tightly coupled, and different parameters
exhibit highly non-uniform perceptual sensitivity, which
makes the choice of embedding domain central to imper-
ceptibility and payload (Xie et al., 2024). Finally, real-time
rendering interfaces enable querying rendered views, en-
abling adaptive attacks such as multi-view probing and op-
timization against the detector (Cayre et al., 2005). These
properties call for explicit and rigorous security specifica-
tions, rather than evaluation regimes that implicitly assume
static media and non-adaptive post-processing.
Despite the growing interest in 3DGS watermarking, a con-
cerning pattern in the R&D efforts of this area is the lack
of systematic and articulated security objectives and real-
istic threat models. Many works leave adversarial access
largely unspecified and unjustified. They also treat the set-
ting as either model-level or rendering-output-level without
defining a taxonomy for concrete deployment scenarios,
and they omit key management. In addition, several sys-
tems adopt the assumption of making a watermark detector
available, such as HiDDeN (Zhu et al., 2018), overlooking
lessons learned from the past literature and industry prac-
tices decades ago. These gaps make it difficult to compare
methods under shared assumptions and leave security claims
hard to reproduce, validate, or justify.
Noting these security weaknesses and limitations, we ad-
vocate a scenario-driven blueprint for 3DGS watermarking
where the threat model should be defined by the deploy-
ment scenario rather than by the media type alone. First of
all, the application and security objectives should be explic-
itly stated, since the success criteria, acceptable costs, and
tradeoffs can vary significantly and are closely associated
with the needs of practical applications. Second, the threat
model should be made explicit within the chosen scenario.
By using an access vector to formalize what the adversary
can access and what capabilities it has, the threat levels
can be described as subsets or clusters of access vectors to
enable systematic evaluations and quantitative comparisons.
Third, algorithm design should match the selected threat
model, with particular attention to keying, detector availabil-
ity, and the implied security boundary. Evaluation should
follow a unified protocol that measures both effectiveness
and risk. We emphasize that no single watermarking
framework can be perfect for all scenarios. Without
scenario-grounded definitions, watermarking applica-
tions can easily collapse into engineering-oriented data
hiding and fail to address the core security questions or
achieve the intended protection objective.
The remainder of this paper is organized as follows. Sec-
tion 2 provides preliminaries on 3DGS and the definitions
for threat modeling. Section 3 analyzes important scenarios
with the defined security system, with a focus on forensic
watermarking. Section 4 builds a reference system and re-
views representative methods with a structured discussion.
Section 5 presents a spread-spectrum baseline allowing for
explicit employment of a security key and experimental re-
sults to illustrate the important trade-offs and challenges.
Section 6 presents the call to action, outlining critical open
questions for the technical community to investigate. Sec-
tion 7 concludes the paper.
2. Preliminaries and Notation
3D Gaussian Splatting.
The primitive of 3DGS is re-
ferred to as a Gaussian, denoted as gi (Kerbl et al., 2023).
Accordingly, a 3DGS scene M is represented as a set of
anisotropic 3D Gaussians:
M = {gi}N
i=1,
gi = {µi, Σi, di, fi},
(1)
where µi ∈R3 specifies the coordinates of the Gaussian
center, Σi ∈SPD(3) is a positive definite covariance matrix
encoding the anisotropic shape, di ∈[0, 1] denotes the
opacity, and fi ∈Rd parameterizes the spherical-harmonic
coefficients for view-dependent color. For any 3D location
x ∈R3, the unnormalized density of the i-th Gaussian is
Gi(x) = exp

−1
2(x −µi)⊤Σ−1
i (x −µi)

.
(2)
Given a viewpoint v, the renderer maps the 3DGS scene M
to a rendered image Iv. For a pixel p on the image plane,
N(p) denotes the set of Gaussians whose splats overlap p.
Each Gaussian gi contributes an RGB color ci(p) ∈R3 and
an opacity weight αi(p) ∈[0, 1]. The rendered pixel color
is obtained by standard front-to-back compositing:
C(p) =
X
i∈N (p)
ci(p) αi(p)
Y
j∈N (p), j≺i
 1 −αj(p)

, (3)
where j ≺i represents that splat j is in front of splat i along
the viewing ray corresponding to pixel p. Equivalently,
defining the transmittance
Ti(p) =
Y
j∈N (p), j≺i
 1 −αj(p)

,
(4)
then we have C(p) =
P
i∈N (p)
ci(p) αi(p) Ti(p). Finally, the
overall rendering process is represented as
Iv = R(M, v; θR),
(5)
where R is the renderer, v represents the viewpoint, and θR
denotes the parameters of the renderer.
2

<!-- page 3 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Figure 1. An authenticated watermarking system. It consists of three parts, setup, embedding, and verification. It is essential to ensure
that the key is transmitted directly from the setup side to the verification portal via a secure channel.
The Authenticated Watermarking System.
As shown in
Figure 1, an authenticated watermarking system for 3DGS
generally consists of three stages: setup, embedding, and
verification (Wu & Liu, 2003a;b).
In the setup stage, the system specifiesa cryptographically
secure key K and its usage, such as how K will be invoked
during deployment. And a copy of K is stored at the verifi-
cation portal via a secure channel. Meanwhile, the system
sets up the watermarking signal detectors and the rendering
viewer at the verification portal. In particular, two types of
detectors should be considered, a 3DGS model-level detec-
tor DM for 3DGS models and an image/video-level detector
DI for rendered images and videos with Iv (Zhao et al.,
2024; M¨uller et al., 2025).
In the embedding stage, the watermark embedder E inserts
a message m ∈{0, 1}L into an original 3DGS model M to
produce a watermarked model Mw:
E(M, m, K; θE) = Mw := {g(w)
i
}Nw
i=1,
(6)
where θE are the parameters of the embedding algorithm.
Generally, Nw ≤N since some watermarking algorithms
may prune the set of Gaussians (Jang et al., 2025).
In verification, an independent verification portal provides
limited query access for watermark verification (Cayre et al.,
2005; Quiring & Rieck, 2018). Given a suspect 3D model
or rendered images, it can output a detection decision and
the extracted watermarking message:
DM(Mw, K; θD) = (ˆy, ˆm, S),
DI({Ivt}T
t=1, K; θD) = (ˆy, ˆm, S),
(7)
where θD denotes the parameters of the detector, ˆy ∈{0, 1}
indicates watermark presence, ˆm is the extracted water-
marking message, and S denotes evaluation metrics, such
as PSNR and SSIM. The portal is not public and only grants
authenticated users a limited number of verification queries
to prevent adaptive query attacks (Cox et al., 1997).
Security Evaluation Model.
Given the 3DGS representa-
tion and the authenticated watermarking system, we define
the following access vector to characterize the privileges of
an adversary: A =

Access-M, Access-Mw, Access-E,
Access-D, Oracle-D, Oracle-R, Key-K

, where Access
indicates full access to the corresponding artifact, including
model files or implementation details, and Oracle indicates
query access provided by the verification portal. For each
entry, the value is set to 1 if the adversary has the specified
full or oracle access, and to 0 otherwise. Unless otherwise
specified, the secret key is unknown to the adversary and
the original model is not directly accessible, which means
that Key-K = Access-M = 0.
Based on the formulation of access vector A, we fur-
ther categorize adversarial settings into three box regimes:
black/grey/white box. It can reflect different risk levels
and is instantiated by specific patterns of entries in A.
The access vectors corresponding to the three box regimes
are denoted as Abb, Agb, and Awb.
In the black-box
regime, the adversary has no direct access to internal ar-
tifacts and can only interact with the system through or-
acle queries. Classical attacks include submitting inputs
to the rendering oracle Oracle-R to obtain rendered im-
ages or videos. These access vectors are typically in the
form of: Abb =

Access-M = 0, Access-Mw = 0,
Access-E = 0, Access-D = 0, Oracle-D, Oracle-R,
Key-K = 0

.
In the white-box regime, the adversary can fully access
the watermarked model and the details of the embedding
and detection algorithms, while the original model and the
key remain unavailable. Accordingly, we have: Awb =

Access-M = 0, Access-Mw = 1, Access-E = 1,
3

<!-- page 4 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Access-D = 1, Oracle-D, Oracle-R, Key-K = 0

,
where the oracle entries Oracle-R and Oracle-D may addi-
tionally be 0 or 1 depending on the deployment.
The grey-box regime refers to any intermediate access pat-
tern that is strictly stronger than pure oracle access but
does not reach the full-access conditions of the white-
box regime.
Then, the grey-box settings satisfy Agb :
Abb ≺Agb ≺Awb. We group all the potential attacks
for 3DGS into three levels according to where the adversary
operates.
(i) 3DGS-level attacks: the adversary directly manipulates
the 3DGS representation, such as editing Gaussian param-
eters, pruning, resampling (Chen et al., 2025; Jang et al.,
2025; Huang et al., 2024).
(ii) Image/video-level attacks: the adversary only accesses
rendered outputs and post-processes them, such as compres-
sion, cropping, resizing, frame dropping, re-encoding, and
screen-recording (Liang et al., 2025).
(iii) Neural-network-level attacks: the adversary exploits a
public or queryable detector and uses learning optimization
to erase or spoof watermarking signals, such as gradient-
based removal, surrogate modeling, and adversarial opti-
mization (Cheng et al., 2024; Liu & Zhang, 2025).
3. Important Use-Case Scenario: Embedded
Fingerprinting / Forensic Watermarking
To motivate scenario-driven modeling for 3DGS watermark-
ing, we shall delineate deployment settings. While water-
marks have been proposed to facilitate tampering detection
and copyright management (Hartung & Kutter, 2002; Wu &
Liu, 2003b). Embedded fingerprinting, also known as Hol-
lywood forensic watermarking, enables tracing individual
copies of audio-visual assets and remains the primary suc-
cessful real-world watermarking application with sustained,
broad deployment (Munoz & Day, 2004; MovieLabs, 2024).
Thus, we use forensic watermarking as the reference setting
for security and adversary analysis.
Another typical scenario, Copyright Ownership Verification,
is discussed in Appendix A. Here, the watermark serves
as a copyright indicator. For each scenario, we (i) define
the overall access vector and the primary threat models,
(ii) specify representative sub-scenarios with access vectors
under black/grey/white-box regimes, respectively, and (iii)
summarize keying mechanisms from traditional media that
are effective under analogous assumptions.
3.1. Black-box Regime
The access vector for this regime can be summa-
rized as A = [0, 0, 0, 0, Oracle-D, Oracle-R, 0], with
Oracle-D, Oracle-R ∈{0, 1}.
In this setting, attacks
rely on observable outputs, namely the rendered images
or videos, and interactions with online rendering or tracking
interfaces.
The common sub-scenarios in this regime include:
(i) Cloud restreaming: this refers to capturing rendered
frames from a 3D asset or rebroadcasting screen record-
ings of the rendering, whereby attribution of the owner
or copyright information is intentionally blurred. Its
access vector is [0, 0, 0, 0, 0, 1, 0].
(ii) Passive leakage of the rendered fixed-view im-
ages/videos: this means that only a fixed-trajectory
video or a small set of screenshots is available. The
corresponding access vector is [0, 0, 0, 0, 0, 0, 0], rep-
resenting the least powerful adversary setting for fin-
gerprinting.
(iii) Using a tracing portal: this refers to a setting where
processed excerpts, such as rendered visual segments,
possibly after compression and/or cropping, are repeat-
edly submitted to an available detector. The returned
portal feedback is then used in subsequent iterative
probing until the forensic watermark may be attenu-
ated so that its tracing outcome becomes unreliable.
The corresponding access vector is [0, 0, 0, 0, 1, 0, 0].
In 3D assets, the verifier often observes only rendered out-
puts, either a multi-frame video along a camera trajectory or
a set of views. Because watermark evidence in the rendering
domain can be weak or view-dependent, keyed aggregation
enables accumulating per-frame or segment statistics into
a confidence-scored suspect set (Cox et al., 1997). In ad-
dition, the key can be used in a challenge-response way
to determine what viewpoints to render or what to select
from a server-specified set of viewpoints. Therefore, stable
attribution is required under multi-view and multi-segment
challenges, which substantially increases the cost of eva-
sion (Cayre et al., 2005).
3.2. White-box Regime
Under the white-box regime for forensic watermarking, the
access vector is A = [0, 1, 1, 1, 1, 1, 0]. In this setting, the
adversary has the internal implementations and the weights
of the detector.
Once the detection mechanism and configuration are known,
evasion can be performed by directly optimizing against
the tracing objective with minimal distortion. Samples
can even be constructed to falsely implicate an innocent
party (Memon & Wong, 2001). Compared with ownership
verification, fingerprinting is more sensitive to the legal
and security consequences of false accusation and fram-
ing. Therefore, the use of a key must remain the root of
trust. Keyed subset selection, projection, permutation, and
thresholds are used so that access to the weights does not
translate into reliable evasion (Cox et al., 1997). In addition,
signatures or message authentication codes can be used to
4

<!-- page 5 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Table 1. Summary of existing 3DGS watermarking methods: embedding design and experimental setup.
Papers
Bit String
Embedding Domain
Embedding Stage
Detector
Datasets
GPU
GuardSplat (Chen et al.,
2025)
16 / 32 / 48 bits
SH Parameters
Post-hoc Optimization
Private
Blender
LLFF
Single RTX 3090
3D-GSW (Jang et al.,
2025)
32 / 48 / 64 bits
Global
Post-hoc Optimization
Public
Blender
LLFF
Mip-NeRF 360
Single A100
GS-Marker (Li et al.,
2025)
32 bits
Global
Single Forward
Process
Public
Objaverse
Blender
OmniObject3D
8 V100
MarkSplatter (Huang
et al., 2025)
32 / 48 bits
Splatter Images
Single Forward
Process
Private
Objaverse
Google Scanned
Objects
8 V100
Water-GS (Tan et al.,
2024)
48 bits
Global
Post-hoc Optimization
Public
Blender
LLFF
Tanks&Temples
Single RTX 3090
GaussianMarker (Huang
et al., 2024)
48 bits
Global
Post-hoc Optimization
Public
Blender
LLFF
Mip-NeRF 360
Single V100
Table 2. Threat-model coverage and evaluation metrics of recent 3DGS watermarking methods.
Papers
GuardSplat
(Chen et al.
2025)
3D-GSW
(Jang et al.
2025)
GS-Marker
(Li et al.
2025)
MarkSplatter
(Huang et al.
2025)
Water-GS
(Tan et al.
2024)
GaussianMarker
(Huang et al.
2024)
Attacking Domains
Rendered
Images
Rendered
Images +
3DGS model
Rendered
Images +
3DGS model
Rendered
Images +
3DGS model
3DGS model
Rendered
Images +
3DGS model
Threat
Models
Gaussian Noise (σ = 0.1)
✓
✓
✓
✓
✓
Rotation (±π/6)
✓
✓
✓
✓
Scaling (75%)
✓
✓
✓
✓
Gaussian blur (σ = 0.1)
✓
✓
✓
✓
2D Cropping (40%)
✓
✓
✓
✓
Brightness (0.5 ∼1.5)
✓
JPEG Compression (Q = 50%)
✓
✓
✓
✓
✓
Translation (20%)
✓
VAE attack
✓
Gaussian noise (σ = 0.1)
✓
✓
✓
Dropout (20%)
✓
✓
✓
✓
3D Cropping (0.5)
✓
✓
✓
✓
✓
Cloning (20%)
✓
Translation (20%)
✓
✓
Evaluation Metrics
Bit Accuracy
Fidelity
Efficiency
Bit Accuracy
Fidelity
Efficiency
Bit Accuracy
Fidelity
Efficiency
Bit Accuracy
Fidelity
Efficiency
Bit Accuracy
Fidelity
Efficiency
Bit Accuracy
Fidelity
Model Distortion
bind claims and reduce the risk of framing (Rivest et al.,
1978). In practice, the objective is set to emphasize a strong
evidentiary chain, a high cost of framing, and auditable
re-verification (Memon & Wong, 2001).
3.3. Grey-box Regime
In this grey-box regime,
the access vector is ex-
pressed as A = [0, 1, 0, 0, Oracle-D, Oracle-R, 0] with
Oracle-D, Oracle-R ∈{0, 1}. One or more multiple per-
sonalized copies may be obtained by the adversary, enabling
model-level editing, re-optimization, and splicing or mixing
across copies. As a result, most threat models focus on the
3DGS model level.
The offline forensic tracing and oracle-guided model evasion
scenarios are typical cases:
(i) Offline forensic tracing includes the ability to track le-
gitimate recipients, supply-chain or collaborator leaks,
and collusive reconstruction. The access vector is
[0, 1, 1, 0, 0, 1, 0].
(ii) Oracle-guided model evasion means that model manip-
ulation is combined with repeated queries to a tracing
interface, and the feedback from the watermark detec-
tor in the oracle is treated as a cost criterion to support
iterative fingerprint removal. The corresponding access
vector is [0, 1, 1, 0, 1, 0, 0].
For this regime, the key is commonly used for individualized
codeword generation, randomized embedding, or tracing
rules (Tardos, 2008). This design enables the intended re-
cipient as the source of the unintended distribution to be
5

<!-- page 6 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
decoded from the leaked copy of the content (Chor et al.,
1994). It also supports collusion resilience by returning at
least one traitor or a small suspect set (Trappe et al., 2003).
Adversaries’ effectiveness through iterative updates to esti-
mate and remove the watermark via repeated queries of the
watermark detector can be further reduced by introducing
keyed challenges, truncating portal outputs, and auditing
and limiting the number of queries.
4. Alternative Views
4.1. View 1: Engineering-Oriented Robustness
Benchmarks Are Sufficient
This view argues that progress in 3DGS watermark-
ing is driven by standardized benchmarks on robustness,
rather than by developing security models and articulat-
ing deployment-specific assumptions. It advocates that the
community should prioritize a shared suite of attacks and
metrics across perturbations on both the 3DGS model and
visual data. As shown in Table 1 and Table 2, such a suite
allows methods to be compared directly and iterated quickly.
Recent papers on 3DGS watermarking usually report the
bit-level accuracy under common distortions, together with
fidelity metrics such as PSNR, SSIM, and LPIPS (Chen
et al., 2025; Jang et al., 2025; Li et al., 2025; Huang et al.,
2025). These works imply that adopting consistent payload
reporting, such as 32/48/64 bits, and expanding benchmark
coverage are effective ways for reproducibility (Jang et al.,
2025).
We would like to note that a limitation of this view is that
the breadth of robustness benchmark does not necessarily
capture security-relevant capabilities, especially when the
adversary is able to access the detector and perform opti-
mization to circumvent the watermarking. That said, robust-
ness and security are not mutually exclusive, as expanding
the set of robustness metrics considered is beneficial to
serve as a prerequisite for security and to achieve clarity
and reproducibility. Robustness alone is not sufficient for
security. Simply enumerating more threat models does not
by itself make a watermarking framework secure (Craver
et al., 2002; Cox et al., 1997). The implication is that a
benchmark-first agenda may yield strong and comparable
numbers, while protocol choices such as keying, detector
availability, and query controls remain underspecified and
vary across deployments, making the system vulnerable to
attacks (Quiring et al., 2018).
4.2. View 2: Public Detectors Enable Open Evaluation
and Faster Progress
This view argues that public detectors should be treated as
a design choice, as they increase transparency and repro-
ducibility. The proponents of this view advocate that bench-
marking should be done without dependence on proprietary
verification portals. In Table 1, several representative 3DGS
watermarking methods rely on image-level detectors, such
as HiDDeN, for watermark extraction from rendered out-
puts (Jang et al., 2025; Li et al., 2025; Tan et al., 2024;
Huang et al., 2024). Public detectors also support stress
testing. They enable different attacks to be implemented in
a consistent and specified manner across papers, rather than
being approximated through limited oracle access.
However, detector availability also reshapes the threat land-
scape as adversaries can exploit the detector to adaptively
remove or insert watermarks. A real-world example is the
Secure Digital Music Initiative (SDMI) (sdm), which ex-
posed a public detection oracle to mimic a blind detector
inside a digital music player. Despite the lack of unwater-
marked references and the absence of disclosed procedures,
the community repeatedly observed that public detector
access allows attackers to collect input–output pairs and
mount oracle attacks, often sufficing to evade detection
even when the detector is complex (Craver et al., 2001;
Wu et al., 2001). When the detector is public, attackers
can replicate the pipeline and optimize against it through
gradient-based attacks, enabling selective erasure and tar-
geted counterfeiting (Quiring et al., 2018; Tondi et al., 2016).
Therefore, security-oriented deployment scenarios should
carefully avoid reliance on a public detector. If a public
detector is adopted for open evaluation, adaptive attacks
should be explicitly included in the threat model, and use of
the detector should be circumvented.
4.3. View 3: Keying Is Optional in Learning-Based
Watermarking
This view argues that keying is not essential in learning-
based watermarking when the detector is deployed as a
private component (Boenisch, 2021). Third parties cannot
extract the watermark because the decoding function and its
parameters are not publicly executable (Cox et al., 1997).
Operational overhead is reduced, as key generation, secure
distribution, storage, and rotation are no longer required.
Without an explicit key, the watermarking scheme reduces
to “security by obscurity”, with the strength of protection
hinging on keeping the detector confidential. This is con-
trary to a well-known security principle and makes both
the threat model and the evaluation setup harder to define
and reproduce (Craver et al., 2002). If the detector weights
are leaked or reverse engineered, the attacker can directly
optimize against the detection objective (Carlini & Wag-
ner, 2017). Even without direct leakage, repeated interac-
tions may allow adversaries to approximate the detector, en-
abling adaptive removal and forgery attacks. Consequently,
if stronger security objectives are targeted, explicit keys and
keyed mechanisms should be treated as necessary protocol
6

<!-- page 7 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Figure 2. Qualitative fidelity under different spread-spectrum embedding amplitudes. We visualize rendered images of the original and
watermarked 3DGS scenes, together with the corresponding difference images (×10), under payload B = 32 and three embedding
strengths α ∈{0.1, 0.01, 0.001}. Results are shown for three scenes, LLFF fern, Mip-NeRF 360 bicycle, and Blender hotdog.
components rather than optional implementation details.
5. Classical Spread-Spectrum Embedding
Existing 3DGS watermarking works also lack a unified
baseline that makes keying choices explicit and experiments
reproducible. To address this gap, we apply the classical
spread-spectrum watermarking baseline to the native 3DGS
parameter space (Cox et al., 1997). This formulation pro-
vides three concrete benefits: (i) a transparent and fully
reproducible embedding/detection pipeline; (ii) a clearly
defined key mechanism, including carrier selection, claim-
to-message binding, and pseudo-random spreading code
generation; and (iii) two controllable trade-off protocols
that enable subsequent work to be compared and iterated
within a consistent coordinate system.
In particular, we represent the embedded message as a B-
bit {±1} payload vector and embed it via code-division
multiplexing (CDM) (Wu et al., 2003) over a reproducible
transform-domain carrier pool. We instantiate this carrier
pool directly in native 3DGS parameters and make its organi-
zation explicitly key-dependent. Therefore, the embedding
layout is both unpredictable to an attacker and exactly re-
producible for a verifier. A second key binds the claim to
the payload, and a third key generates the pseudo-random
spreading codes used for multi-bit superposition.
On the detection side, we follow a classical non-blind proto-
col. Given the original model and a suspect model, the detec-
tor forms a transform-domain residual over the same carrier
pool and recovers each bit by correlation decoding followed
by a sign test, reporting bit accuracy as the primary outcome.
Moreover, we adopt two power-normalization strategies to
expose the classical “capacity–robustness–fidelity” trade-
offs on standard 3DGS benchmarks. Full implementation
details are provided in Appendix B.
In the experiment, we evaluate the proposed spread-
spectrum embedding and the non-blind detection baseline
on three standard benchmarks, Blender (Mildenhall et al.,
2021), LLFF (Mildenhall et al., 2019), and Mip-NeRF
360 (Barron et al., 2022). Overviews of other commonly
used datasets in this area, as well as detailed analyses of rep-
resentative algorithms, are available in Appendix C. These
datasets are widely adopted in the NeRF and 3DGS literature
and cover both synthetic and real-world scene distributions.
In addition, the common payload lengths B ∈{32, 48, 64}
are considered. For the transform-domain processing, we
set the mid-band interval [0.10, 0.18] as the candidate carrier
range for the 1D DCT. The embedding strength parameter α
is set to {0.1, 0.01, 0.001}, forming controlled groups with
different embedding intensities.
For fidelity evaluation, we measure image quality on ren-
dered images and utilize PSNR, SSIM, and LPIPS to evalu-
ate the impact of watermark embedding on rendering fidelity.
For robustness evaluation, detection experiments are con-
ducted under a unified Gaussian-noise attack setting and use
bit accuracy as the detection metric.
Figure 2 compares the original and watermarked rendered
images together with the ×10 difference images on three
scenes with a fixed payload B = 32 under different embed-
ding amplitudes α. As α decreases from 0.1 to 0.01 and
0.001, visible artifacts and rendering residuals shrink, and
the amplified differences transition from strong speckle pat-
terns to localized residuals and then to almost black images.
We also compare the fidelity with different payload lengths,
which is displayed in Figure 4 in Appendix D.
7

<!-- page 8 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Table 3. Quantitative comparisons under different payload sizes and embedding amplitudes.
Embedding Amplitude
32 bits
48 bits
64 bits
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
α = 0.1
25.43
0.825
0.255
23.74
0.784
0.296
22.56
0.752
0.325
α = 0.01
44.96
0.995
0.009
43.28
0.993
0.014
42.09
0.991
0.019
α = 0.001
58.85
0.998
0.001
57.96
0.997
0.002
57.32
0.995
0.004
Figure 3 examines the visual impact of different payload
lengths while fixing the embedding strength at α = 0.01
under the fixed per-bit energy setting. While the renderings
remain visually similar as the payload increases, the fidelity
metrics degrade monotonically, with PSNR and SSIM de-
creasing and LPIPS increasing.
Table 3 reports fidelity metrics under different payload
sizes and embedding amplitudes. The consistent mono-
tonic trends indicate that this spread-spectrum scheme can
still cause measurable fidelity degradation even when the
rendering differences are not visually apparent.
6. Call to Action
We recommend adopting our scenario-driven blueprint as
a minimum disclosure standard for R&D on 3DGS water-
marking. First, each work should state the deployment
scenario and security objective, together with a concrete
success criterion and acceptable costs, since different objec-
tives are associated with different trade-offs. Second, the
threat model should be articulated within the chosen sce-
nario, preferably including an access vector that formalizes
what the adversary can access. Under this formulation, con-
ventional regimes of white/black/gray boxes can be viewed
as subsets of access patterns, enabling more comparable
security statements. Third, the algorithm design should be
aligned with the stated threat model and specify whether the
scheme is keyed. Fourth, evaluation should follow a proto-
col that reports both effectiveness and exposure, rather than
presenting engineering-oriented data hiding results under
implicit default assumptions.
We recognize the significant challenge associated with cross-
domain watermark detection, namely, to detect a watermark
embedded at the 3DGS model level from the rendered im-
ages/videos. 3DGS watermarking spans two coupled do-
mains, the model domain and the rendering domain. An
ideal framework should account for both model-level and
image/video-level detection. However, existing 3DGS wa-
termarking pipelines rarely design detectors specifically for
2D rendered outputs. Instead, many works directly reuse
public detectors from image watermarking, such as HiD-
DeN (Zhu et al., 2018). While this reuse reduces engineer-
ing effort, as discussed in this paper, prior literature and
32
48
64
Payload (bits)
65%
70%
75%
80%
85%
90%
95%
100%
Bit Accuracy
0.817
0.756
0.737
0.867
0.833
0.801
0.790
0.717
0.703
Blender
Mip-NeRF 360
LLFF
Figure 3. The trade-off between bit accuracy and payload length.
Bit accuracy is represented as a function of payload length B ∈
{32, 48, 64} under a fixed-total embedding energy constraint.
past practices have shown that a public detector can become
an entry point to be exploited by adversaries (Cayre et al.,
2005). Therefore, we call on the community to embrace
this research challenge, and design and clearly articulate
framework-specific detectors that are tailored to the intended
deployment objective and threat model, rather than default-
ing to generic public detectors.
7. Conclusion
In this position paper, we argue that the R&D of 3DGS
watermarking is more limited by unclear or questionable
security definitions than by embedding techniques. We
advocate a scenario-driven approach and utilize forensic
watermarking scenarios as examples to demonstrate how to
articulate the threat models and analyze the solutions. We
incorporate the lessons learned from past literature and in-
dustry practices of audio-visual watermarking and construct
a reference system to show how existing design choices im-
plicitly assume choices of threat and adversary models. In
addition, we provide a reproducible spread-spectrum base-
line, in which keying mechanisms are made explicit and
trade-off protocols are standardized. We hope these dis-
cussions help move 3DGS watermarking from ad-hoc data
hiding toward scenario-grounded, comparable, and verifi-
able security.
8

<!-- page 9 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
References
Secure Digital Music Initiative (SDMI) 2000-2001. URL
https://en.wikipedia.org/wiki/Secure_
Digital_Music_Initiative.
Bagdasarian, M. T., Knoll, P., Li, Y., Barthel, F., Hilsmann,
A., Eisert, P., and Morgenstern, W. 3DGS.zip: A survey
on 3D Gaussian Splatting compression methods. In Com-
puter Graphics Forum, volume 44, pp. e70078. Wiley
Online Library, 2025.
Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P.,
and Hedman, P. Mip-NeRF 360: Unbounded anti-aliased
neural radiance fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pp. 5470–5479, 2022.
Boenisch, F. A systematic review on model watermarking
for neural networks. Frontiers in Big Data, 4:729663,
2021.
Carlini, N. and Wagner, D. Towards evaluating the robust-
ness of neural networks. In 2017 IEEE Symposium on
Security and Privacy (SP), pp. 39–57. IEEE, 2017.
Cayre, F., Fontaine, C., and Furon, T. Watermarking secu-
rity: theory and practice. IEEE Transactions on Signal
Processing, 53(10):3976–3987, 2005.
Chen, Z., Wang, G., Zhu, J., Lai, J., and Xie, X. GuardSplat:
Efficient and robust watermarking for 3D Gaussian Splat-
ting. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 16325–16335, 2025.
Cheng, Z., Liu, Z., Guo, T., Feng, S., Liu, D., Tang, M.,
and Zhang, X. Badpart: Unified black-box adversarial
patch attacks against pixel-wise regression tasks. arXiv
preprint arXiv:2404.00924, 2024.
Chor, B., Fiat, A., and Naor, M. Tracing traitors. In An-
nual International Cryptology Conference, pp. 257–270.
Springer, 1994.
Cox, I. J., Kilian, J., Leighton, F. T., and Shamoon, T.
Secure spread spectrum watermarking for multimedia.
IEEE Transactions on Image Processing, 6(12):1673–
1687, 1997.
Cox, I. J., Miller, M. L., Bloom, J. A., Fridrich, J., and
Kalker, T. Digital Watermarking. Morgan Kaufmann
Publishers, 54:56–59, 2008.
Craver, S., Memon, N., Yeo, B.-L., and Yeung, M. M. Re-
solving rightful ownerships with invisible watermarking
techniques: Limitations, attacks, and implications. IEEE
Journal on Selected Areas in Communications, 16(4):
573–586, 2002.
Craver, S. A., Wu, M., Liu, B., Swartzlander, B., Wallach,
D. S., Dean, D., and Felten, E. W. Reading between the
lines: Lessons from the SDMI challenge. In 10th USENIX
Security Symposium (USENIX Security 01), 2001.
Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel,
O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi,
A., and Farhadi, A. Objaverse: A universe of annotated
3D objects. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 13142–
13153, 2023.
Dengel, A., Iqbal, M. Z., Grafe, S., and Mangina, E. A
review on augmented reality authoring toolkits for educa-
tion. Frontiers in Virtual Reality, 3:798032, 2022.
Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman,
R., Reymann, K., McHugh, T. B., and Vanhoucke, V.
Google scanned objects: A high-quality dataset of 3D
scanned household items. In 2022 International Confer-
ence on Robotics and Automation (ICRA), pp. 2553–2560.
IEEE, 2022.
Guo, J., Xin, Y., Liu, G., Xu, K., Liu, L., and Hu, R. Ar-
ticulatedgs: Self-supervised digital twin modeling of ar-
ticulated objects using 3D Gaussian Splatting. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pp. 27144–27153, 2025.
Hartung, F. and Kutter, M. Multimedia watermarking tech-
niques. Proceedings of the IEEE, 87(7):1079–1107, 2002.
Huang, X., Li, R., Cheung, Y.-m., Cheung, K. C., See,
S., and Wan, R. Gaussianmarker: Uncertainty-aware
copyright protection of 3D Gaussian Splatting. Advances
in Neural Information Processing Systems, 37:33037–
33060, 2024.
Huang, X., Luo, Z., Song, Q., Wang, R., and Wan, R. Mark-
Splatter: Generalizable watermarking for 3D Gaussian
Splatting model via splatter image structure.
In Pro-
ceedings of the 33rd ACM International Conference on
Multimedia, pp. 12189–12198, 2025.
Jang, Y., Park, H., Yang, F., Ko, H., Choo, E., and Kim, S.
3D-GSW: 3D Gaussian Splatting for robust watermark-
ing. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 5938–5948, 2025.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3D
Gaussian Splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
Knapitsch, A., Park, J., Zhou, Q.-Y., and Koltun, V. Tanks
and Temples: Benchmarking large-scale scene reconstruc-
tion. ACM Transactions on Graphics (ToG), 36(4):1–13,
2017.
9

<!-- page 10 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Li, L., Wang, J., Ming, X., and Lu, Y. GS-Marker: Gener-
alizable and robust watermarking for 3D Gaussian Splat-
ting. arXiv preprint arXiv:2503.18718, 2025.
Liang, X., Liu, G., Si, Y., Hu, X., and Qian, Z. Screen-
Mark: Watermarking arbitrary visual content on screen.
In Proceedings of the AAAI Conference on Artificial In-
telligence, volume 39, pp. 26273–26280, 2025.
Liu, Z. and Zhang, H. Stealthy backdoor attack in self-
supervised learning vision encoders for large vision lan-
guage models. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pp. 25060–25070,
2025.
Memon, N. and Wong, P. W. A buyer-seller watermarking
protocol. IEEE Transactions on Image Processing, 10(4):
643–649, 2001.
Mildenhall, B., Srinivasan, P. P., Ortiz-Cayon, R., Kalantari,
N. K., Ramamoorthi, R., Ng, R., and Kar, A. Local light
field fusion: Practical view synthesis with prescriptive
sampling guidelines. ACM Transactions on Graphics
(ToG), 38(4):1–14, 2019.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. NeRF: Representing scenes
as neural radiance fields for view synthesis. Communica-
tions of the ACM, 65(1):99–106, 2021.
MovieLabs.
Movielabs specification for enhanced
content protection — version 1.4.
Technical re-
port,
Motion
Picture
Laboratories,
Inc.,
August
2024. URL https://movielabs.com/ngvideo/
MovieLabs_ECP_Spec_v1.4.pdf.
Accessed:
2026-01-27.
M¨uller, A., Lukovnikov, D., Thietke, J., Fischer, A., and
Quiring, E. Black-box forgery attacks on semantic wa-
termarks for diffusion models. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pp.
20937–20946, 2025.
Munoz, L. and Day, P.
Suspected movie pirate ar-
rested.
Los Angeles Times, January 2004.
URL
https://www.latimes.com/archives/
la-xpm-2004-jan-23-fi-arrest23-story.
html.
Peng, Z., Shao, T., Liu, Y., Zhou, J., Yang, Y., Wang, J.,
and Zhou, K. Rtg-slam: Real-time 3D reconstruction
at scale using Gaussian Splatting. In ACM SIGGRAPH
2024 Conference Papers, pp. 1–11, 2024.
Quiring, E. and Rieck, K. Adversarial machine learning
against digital watermarking. In 2018 26th European
Signal Processing Conference (EUSIPCO), pp. 519–523.
IEEE, 2018.
Quiring, E., Arp, D., and Rieck, K. Forgotten siblings:
Unifying attacks on machine learning and digital water-
marking. In 2018 IEEE European Symposium on Security
and Privacy (EuroS&P), pp. 488–502. IEEE, 2018.
Rivest, R. L., Shamir, A., and Adleman, L. A method for
obtaining digital signatures and public-key cryptosystems.
Communications of the ACM, 21(2):120–126, 1978.
Stamm, M. C., Wu, M., and Liu, K. R. Information forensics:
An overview of the first decade. IEEE Access, 1:167–200,
2013.
Tan, Y., Liu, X., Xie, S., Chen, B., Xia, S.-T., and Wang,
Z.
WATER-GS: Toward copyright protection for 3D
Gaussian Splatting via universal watermarking. arXiv
preprint arXiv:2412.05695, 2024.
Tardos, G. Optimal probabilistic fingerprint codes. Journal
of the ACM (JACM), 55(2):1–24, 2008.
Tondi, B., Comesa˜na-Alfaro, P., P´erez-Gonz´alez, F., and
Barni, M. Smart detection of line-search oracle attacks.
IEEE Transactions on Information Forensics and Security,
12(3):588–603, 2016.
Trappe, W., Wu, M., Wang, Z. J., and Liu, K. R. Anti-
collusion fingerprinting for multimedia. IEEE Transac-
tions on Signal Processing, 51(4):1069–1087, 2003.
Wegen, O., Scheibel, W., Trapp, M., Richter, R., and Doll-
ner, J.
A survey on non-photorealistic rendering ap-
proaches for point cloud visualization. IEEE Transactions
on Visualization and Computer Graphics, pp. 1–20, 2024.
Wu, M. and Liu, B. Data hiding in image and video. I.
fundamental issues and solutions. IEEE Transactions on
Image Processing, 12(6):685–695, 2003a.
Wu, M. and Liu, B. Multimedia Data Hiding. Springer
Science & Business Media, 2003b.
Wu, M., Craver, S., Felten, E. W., and Liu, B. Analysis
of attacks on SDMI audio watermarks. In 2001 IEEE
International Conference on Acoustics, Speech, and Sig-
nal Processing. Proceedings (Cat. No. 01CH37221), vol-
ume 3, pp. 1369–1372. IEEE, 2001.
Wu, M., Yu, H., and Liu, B. Data hiding in image and video.
II. designs and applications. IEEE Transactions on Image
Processing, 12(6):696–705, 2003.
Wu, T., Zhang, J., Fu, X., Wang, Y., Ren, J., Pan, L., Wu,
W., Yang, L., Wang, J., Qian, C., et al. OmniObject3D:
Large-vocabulary 3D object dataset for realistic percep-
tion, reconstruction and generation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 803–814, 2023.
10

<!-- page 11 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Xie, S., Zhang, W., Tang, C., Bai, Y., Lu, R., Ge, S., and
Wang, Z. Mesongs: Post-training compression of 3D
Gaussians via efficient attribute transformation. In Eu-
ropean Conference on Computer Vision, pp. 434–452.
Springer, 2024.
Zhao, X., Zhang, K., Su, Z., Vasan, S., Grishchenko, I.,
Kruegel, C., Vigna, G., Wang, Y.-X., and Li, L. Invisible
image watermarks are provably removable using gener-
ative AI. Advances in Neural Information Processing
Systems, 37:8643–8672, 2024.
Zhu, J., Kaplan, R., Johnson, J., and Fei-Fei, L. Hidden:
Hiding data with deep networks. In Proceedings of the
European Conference on Computer Vision (ECCV), pp.
657–672, 2018.
11

<!-- page 12 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
A. Additional Use-Case Scenario: Copyright Ownership Verification
Copyright ownership verification refers to the setting where an authorship or ownership dispute arises. In this setting, the
rights holder must demonstrate that a given 3DGS model or its rendered outputs indeed originate from the legitimate owner,
thereby supporting ownership claims, enforcement, and licensing management. The central objective of this scenario is a
verifiable ownership statement.
A.1. Black-box Regime
Under the black-box regime, the access vector can be summarized as A = [0, 0, 0, 0, Oracle-D, Oracle-R, 0], with
Oracle-D, Oracle-R ∈{0, 1}. In this section, the entry ordering of all access vectors follows that of Eq. (8). This setting
indicates that the adversary has no access to Mw and the implementations or weights of the embedder and detector. Attacks
are restricted to interactions with available oracle interfaces, which are mainly concentrated on the image/video-level (Cox
et al., 1997).
Two classical scenarios are considered, interactive viewers and broadcast media:
(i) Interactive viewers means that the adversary can only capture screenshots or screen recordings from an interactive
UI, while the 3DGS model file remains inaccessible. The corresponding access vector is [0, 0, 0, 0, 0, 1, 0]. The attack
surface is primarily on the output side, such as compression artifacts and recording noise, in order to degrade watermark
detectability.
(ii) Broadcast case is that only a fixed-trajectory video or a small set of images is available, without viewpoint control or
repeated interaction. The corresponding access vector is [0, 0, 0, 0, 0, 0, 0], which represents the weakest attack setting
among the above.
In these two scenarios, the key is commonly used for correlation-accumulation detection across multiple views or frames,
where coherent aggregation of weak signals is enabled only with the correct key (Hartung & Kutter, 2002; Cox et al., 1997).
A.2. White-box Regime
For the white-box regime, the access vector can be concluded as A = [0, 1, 1, 1, Oracle-D, 1, 0], with Oracle-D ∈{0, 1}.
This regime assumes that the adversary has access to the implementations and weights of both embedder and detector,
enabling gradient-based attacks (Carlini & Wagner, 2017). Only Key-k and Access-M remain unavailable.
Wo discuss two typical scenarios here, local white-box optimization and online white-box optimization:
(i) Local white-box optimization corresponds to the setting where the adversary, after obtaining the weights of embedder
and detector, directly performs gradient-based white-box optimization to modify Mw, or its rendered images or videos,
in order to erase the watermark with minimal distortion. The corresponding access vector is [0, 1, 1, 1, 0, 1, 0].
(ii) Online white-box optimization corresponds to the setting where white-box backpropagation is available and a portal
can also be queried to validate or align intermediate results, which improves attack efficiency. This access vector is
[0, 1, 1, 1, 1, 1, 0].
White-box leakage is widely regarded as the most challenging setting in practice (Quiring et al., 2018). In this regime, the
key is commonly treated as a means to raise the adversary’s acquisition cost (Cox et al., 1997). It serves as a runtime secret
that controls where to look, how to read, and whether authentication is possible, through keyed selection, projection, and
authentication (Cox et al., 1997). Hence, leaking weights does not directly translate into precise erasure or reliable forgery.
A.3. Grey-box Regime
In the grey-box regime, the access vector can be summarized as A = [0, 1, 0, 0, Oracle-D, Oracle-R, 0], with
Oracle-D, Oracle-R ∈{0, 1}. In this regime, the adversary can access Mw and therefore directly manipulate 3DGS
parameters. Meanwhile, Access-D = 0 holds, and the detector weights remain inaccessible, which prevents white-box
backpropagation through the detector. Accordingly, the main threat models concentrate at the 3DGS model level, including
parameter perturbation, Gaussian pruning, and retraining or re-optimization.
Two canonical scenarios are considered in this regime, offline piracy with model resale and online piracy with model resale:
(i) Offline piracy with model resale refers to the case where a pirate acquires Mw and performs offline modifications,
such as pruning, re-optimization, or perturbation, aiming to suppress the watermark before resale. The corresponding
12

<!-- page 13 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
access vector is [0, 1, 1, 0, 0, 1, 0]. Since Oracle-D = 0 during the attack process, the attack is inherently blind.
(ii) Online piracy with model resale corresponds to the case where a pirate acquires Mw and can repeatedly query a
verification portal, using the returned feedback to iteratively update the modified model until detection fails. The
corresponding access vector is [0, 1, 1, 0, 1, 1, 0].
In both scenarios, the key is commonly used to determine the carrier subset and the bit mapping, through keyed carrier
selection and keyed mapping (Cox et al., 1997). This design makes precise erasure more difficult and forces broader model
degradation in order to reliably remove the watermark (Craver et al., 2002).
B. The Details of Spread Spectrum Algorithm
B.1. Keying Mechanism
Based on the Kerckhoffs assumption, the embedding algorithm should be treated as public, while security and reproducibility
are governed by secret keys. Let the claim be a string C and the payload length be B. Three keys are introduced, Ksel,
Kcode, and Kseq. The required randomness is derived from an HMAC-based random function PRF(K, τ) ∈{0, 1}ℓ. Using
Kcode, the claim is mapped into a bipolar payload
b(C) = 2 · PRF(Kcode, C)1:B −1B ∈{−1, +1}B,
where PRF(·)1:B denotes taking the first B bits. This construction makes it difficult to produce the correct payload for the
same claim without Kcode, while ensuring exact reproducibility under the same key.
The carrier-organization key Ksel makes the embedding layout key-dependent and reproducible. Let the transform-domain
candidate carrier pool be P = {(p, k)} with size T. The Ksel can generate a random permutation over carrier indices
π ←Permute
 PRF(Ksel, C ∥perm), T

,
where ∥denotes concatenation and perm is a fixed context tag used only to separate purposes. This step preserves the
carrier set P and globally mixes carriers across fields and frequency indices, removing predictable structural ordering.
The spreading key generates a T-length random bipolar template for each bit position, used for both embedding superposition
and correlation decoding. For each j ∈{1, . . . , B}, we define
sj(C) = 2 · PRF(Kseq, C ∥j)1:T −1T ∈{−1, +1}T .
The detector reproduces sj(C) using the same Kseq and correlates it with the observed residual vector to recover the bit sign.
Even if an adversary knows the overall embedding domain and how candidates are constructed, without Kseq it remains
difficult to reconstruct the bit templates required for decoding or matched removal.
B.2. Spread Spectrum Embedding Algorithm
In our baseline experimental setting, the spherical-harmonics parameters of 3DGS are selected as the embedding domain.
Compared with other Gaussian parameters, center coordinates with dimension 3, opacity with dimension 1, rotation with
dimension 4, and scale with dimension 3, the SH parameters provide a 48-dimensional embedding space. Moreover, other
Gaussian parameters are tightly coupled with scene geometry, which means that small perturbations can severely degrade
the overall scene fidelity. Therefore, restricting modifications to the SH parameters is more favorable, and is also a common
choice in the existing works.
For each scene, all SH parameters of all Gaussians are concatenated into a sequence of 1D length, and a 1D DCT is applied
along the Gaussian index axis to obtain transform-domain coefficients. We then select a mid-band interval of DCT frequency
indices as candidate embedding locations to form the transform-domain candidate carrier pool P, and denote its size by
T = |P|. To make the embedding layout key-dependent and reproducible, Ksel is used to generate a keyed permutation π
over carrier indices.
Using Kcode, a bipolar payload vector b(C) ∈{−1, +1}B is derived. For each bit position j ∈{1, . . . , B}, a length-T
bipolar spreading template sj(C) ∈{−1, +1}T is generated using Kseq. Following that setting, we adopt a code-division
superposition embedding in the carrier domain and define the overall perturbation vector as
δ(C) =
B
X
j=1
αj bj(C) sj(C),
13

<!-- page 14 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
where αj denotes the per-bit embedding amplitude. Finally, after injecting the perturbation, the updated transform-domain
representation is mapped back to the parameter domain, producing the watermarked model Mw.
B.3. 3DGS Model Watermark Detector
On the 3DGS-level detection side, we follow the classical non-blind detection protocol widely used in traditional media.
Given access to the original model M and a suspect model f
M, we construct the transform-domain residual by taking
DCT coefficient differences at carrier locations in P. Using Ksel and the claim C, the detector reproduces the same keyed
permutation π. We further adopt a detector-side carrier budget by operating on a deterministic subset Ω⊆{1, . . . , T} of
carrier indices in order to reduce computational cost and standardize the detection overhead. Specifically, Ωis sampled from
the permuted carrier order in a key-dependent and reproducible manner using Ksel and the claim C.
For each bit position j, the detector generates the spreading template sj(C) using Kseq, extracts its subvector sj,Ω(C), and
then computes the correlation score
Sj = ⟨zΩ, sj,Ω(C)⟩,
where zΩis the residual observation vector extracted according to the permuted carrier order and the budgeted subset.
Finally, the bit can be recovered via a sign test
ˆbj =
(
+1,
Sj ≥0,
−1,
Sj < 0.
C. Analysis of the Existing Works on 3DGS watermarking
In Table 1, we decompose representative existing works and summarize them along several axes. Bit String indicates
the information payload carried by the watermark (in bits). It directly governs two core trade-offs, payload–robustness
and payload–fidelity. In this paper, we advocate reporting results under 32/48/64-bit payloads. Too short payloads may
only support coarse identifiers and can be insufficient for richer claims, making them less informative in practice (Chen
et al., 2025). Moreover, reporting a single payload length can mislead subsequent readers about the flexibility of payload
configuration and obscure the capacity-related trade-offs (Li et al., 2025; Tan et al., 2024; Huang et al., 2024).
For the Embedding Domain, gradient-based optimization in neural rendering can preserve the fidelity of the watermarked
model, which motivates many methods to embed watermarks globally across the representation (Jang et al., 2025; Li et al.,
2025; Tan et al., 2024; Huang et al., 2024). However, it is crucial to note that when watermark embedding is performed with
some computing methods, certain Gaussian parameters, such as positions and orientations, are coupled with scene geometry.
Even small perturbations to these geometry-sensitive parameters can cause disproportionate geometric artifacts and visible
distortions.
For the Embedding Stage, post-hoc optimization is among the most widely adopted and stable paradigms, including in
NeRF-based watermarking. Nevertheless, it typically incurs substantial optimization time. In contrast, single-forward
schemes avoid iterative optimization, but often rely on cross-domain transformations, such as from 3DGS parameters to
intermediate images representations, which can introduce information loss. (Huang et al., 2025). Existing 3DGS-level
single-forward schemes still remain less mature in robustness and security, leaving substantial room for improvement (Li
et al., 2025).
The detector design for rendered images raises a fundamental security concern. Several works adopt HiDDeN (Zhu et al.,
2018) as the detector for 2D rendered watermarks (Jang et al., 2025; Li et al., 2025; Tan et al., 2024; Huang et al., 2024),
implying that detection is publicly runnable. Meanwhile, none of these works explicitly incorporate constraints such as
secret keys or private parameters, which exposes them to standard security risks such as reverse engineering, adaptive
querying, and forgery. In particular, if the detector is public, an attacker can replicate the pipeline and mount gradient-based
attacks to selectively erase or counterfeit the watermark.
The datasets used in prior studies can be broadly categorized into three groups: synthetic scenes, real-world scenes, and
large-scale object-centric data. Synthetic scenes like Blender (Mildenhall et al., 2021) offer controlled factors and aligned
quantitative evaluation, but can exhibit distributional bias and under-represent real capture artifacts. Real-world scenes
contain LLFF (Mildenhall et al., 2019), Mip-NeRF 360 (Barron et al., 2022), and Tanks&Temples (Knapitsch et al., 2017),
which better reflect realistic capture noise and complex appearance, but make training and rendering settings harder to
14

<!-- page 15 -->
Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit
Figure 4. Qualitative fidelity with different payload lengths. Rendered original, watermarked, and difference (×10) images are represented
with a fixed embedding strength α = 0.01 and payload lengths B ∈{32, 48, 64}. The three rows correspond to LLFF fern, Mip-
NeRF 360 bicycle, and Blender hotdog, respectively.
standardize across methods. Large-scale object-centric datasets have Objaverse (Deitke et al., 2023), OmniObject3D (Wu
et al., 2023), and Google Scanned Objects (Downs et al., 2022). They emphasize asset diversity and large-scale generalization,
which are suitable for stress-testing stability across many objects and categories. However, most samples depict isolated
objects with simple or missing backgrounds, lacking global illumination, occlusions, complex backgrounds, and inter-object
light transport effects commonly present in real scenes.
In Table 2, the threat models simulated by all the aforementioned works are listed. We use upright font for attacks applied
to rendered 2D media, such as Gaussian Noise (σ = 0.1), and italic font for attacks applied directly to the 3DGS model,
such as Gaussian noise (σ = 0.1). For each attack, the attacking domain and the corresponding evaluation metrics are also
specified.
D. Trade-off Between Fidelity and Different Payload Lengths
Figure 4 illustrates the trade-off between the robustness and payload under a fixed total embedding energy constraint using
additive Gaussian noise with σ = 0.3. Detection accuracy decreases as the payload increases, since a fixed total embedding
energy budget allocates less energy to each bit, which in turn weakens robustness under noise.
15
