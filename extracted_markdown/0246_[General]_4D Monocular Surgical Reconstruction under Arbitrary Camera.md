<!-- page 1 -->
4D Monocular Surgical Reconstruction under Arbitrary Camera Motions
Jiwei Shana,b,c, Zeyu Caid, Cheng-Tai Hsiehe, Yirui Lia, Hao Liub,c, Lijun Han∗e, Hesheng Wange,f, Shing Shin Cheng∗a
aDepartment of Mechanical and Automation Engineering and T Stone Robotics Institute, The Chinese University of Hong Kong, Hong Kong.
bShenyang Institute of Automation, Chinese Academy of Sciences, Shenyang, China
cState Key Laboratory of Robotics and Intelligent Systems, Shenyang, China
dSchool of Integrated Circuits, Shanghai Jiao Tong University, Shanghai, China
eSchool of Automation and Intelligent Sensing, Shanghai Jiao Tong University, Shanghai, China
fKey Laboratory of System Control and Information Processing, Ministry of Education of China, Shanghai, China
Abstract
Reconstructing deformable surgical scenes from endoscopic videos is a challenging task with important clinical applications. Re-
cent state-of-the-art approaches, such as those based on implicit neural representations or 3D Gaussian splatting, have made notable
progress in this area. However, most existing methods are designed for deformable scenes with fixed endoscope viewpoints and rely
on stereo depth priors or accurate structure-from-motion for both initialization and optimization. This limits their ability to handle
monocular sequences with large camera movements, restricting their use in real clinical settings. To address these limitations, we
propose Local-EndoGS, a high-quality 4D reconstruction framework for monocular endoscopic sequences with arbitrary camera
motion. Local-EndoGS introduces a progressive, window-based global scene representation that allocates local deformable scene
representations for each observed window, enabling scalability to long sequences with substantial camera movement. To overcome
unreliable initialization due to the lack of stereo depth or accurate structure-from-motion, we propose a coarse-to-fine initialization
strategy that integrates multi-view geometry, cross-window information, and monocular depth priors, providing a robust foundation
for subsequent optimization. In addition, we incorporate long-range 2D pixel trajectory constraints and physical motion priors
to improve the physical plausibility of the recovered deformations. We comprehensively evaluate Local-EndoGS on three public
endoscopic datasets with deformable scenes and varying camera motions. Local-EndoGS achieves superior performance in both ap-
pearance quality and geometry, consistently outperforming state-of-the-art methods. Extensive ablation studies further validate the
effectiveness of our key designs. Our code will be released upon acceptance at https://github.com/IRMVLab/Local-EndoGS.
Keywords: Endoscopy, 4D Surgical Reconstruction, Monocular, 3d Gaussian Splitting
1. Introduction
Endoscopes are widely used to examine almost all anatom-
ical structures in the human body.
Many diseases can also
be treated with endoscopes equipped with specialized medical
instruments. As endoscopic imaging and computational tech-
nologies advance, high-quality surgical reconstruction from en-
doscopic images is becoming increasingly important in medi-
cal applications. For example, it enables virtual and augmented
reality tools for surgical simulation and training, which help
improve learning and practical skills Ota et al. (1995). High-
quality reconstruction also enhances visualization for diagnosis
and supports accurate preoperative planning. Detailed 3D mod-
els of patient-specific anatomy help surgeons understand com-
plex structures and reduce surgical risks Maier-Hein and et al.
(2017). However, achieving high-quality surgical reconstruc-
tion presents several challenges.
First, physiological move-
ments such as breathing and heartbeat, as well as interactions
between surgical instruments and soft tissue, cause the surgical
∗Corresponding authors
Email addresses: lijun_han@sjtu.edu.cn (Lijun Han∗),
sscheng@cuhk.edu.hk (Shing Shin Cheng∗)
scene to deform. In addition, the confined space within the body
limits the size of endoscopes, which prevents direct acquisition
of depth information. This restriction also limits possible view-
ing angles, resulting in fewer three-dimensional cues from the
surgical scene. To address these challenges, various reconstruc-
tion algorithms have been proposed. These include traditional
methods based on depth estimation or SLAM Schmidt et al.
(2024), methods using implicit neural representations Wang
et al. (2022); Zha et al. (2023); Yang et al. (2024a); Li et al.
(2024b); Shan et al. (2025b); Batlle et al. (2023); Shan et al.
(2024a); Guo et al. (2024b); Shan et al. (2024b), and methods
based on 3D Gaussian splatting Kerbl et al. (2023); Yang et al.
(2024d); Shan et al. (2025a); Li et al. (2024a); Xie et al. (2024);
Liu et al. (2024); Huang et al. (2024b); Liu et al. (2025); Guo
et al. (2024a); Wang et al. (2024a).
Traditional surgical scene reconstruction algorithms have
been extensively developed over the past decades and have
achieved significant progress Schmidt et al. (2024). However,
these methods either assume static conditions or are unable to
effectively capture the complex topological changes associated
with soft tissue deformation Wang et al. (2022). In recent years,
implicit neural methods, which use neural networks to represent
three-dimensional spaces implicitly, have shown improved per-
Preprint submitted to Elsevier
February 20, 2026
arXiv:2602.17473v1  [cs.CV]  19 Feb 2026

<!-- page 2 -->
Figure 1: (a)–(c) Illustrations of three typical types of camera motion in surgi-
cal scenes: (a) fixed camera, (b) camera moving around the tissue, and (c) cam-
era moving forward. (d) Monocular reconstruction results of different methods
under camera motion. State-of-the-art 4D surgical reconstruction algorithms
experience significant degradation in reconstruction quality when the camera
moves, while our method maintains superior performance.
formance in several tasks Tewari et al. (2020, 2022), including
surgical scene reconstruction. Most INR methods model de-
formable surgical scenes using a single canonical space and a
deformation field, as illustrated in Fig. 1. The canonical space
typically represents the three-dimensional state of the scene at
a reference time (usually set as t = 0) and is implicitly modeled
by a multilayer perceptron (MLP). The deformation network
establishes correspondences between the canonical space and
the observed space (t > 0), thereby capturing scene deforma-
tions. During training, the model parameters—including both
the canonical space MLP and the deformation network—are
optimized by minimizing the loss between rendered results and
the input RGB images, as well as stereo depth priors Wang
et al. (2022); Zha et al. (2023); Yang et al. (2024a); Li et al.
(2024b); Shan et al. (2025b, 2024a,b). Compared with tradi-
tional algorithms, INR methods can provide high-quality re-
constructions of deformable scenes and enable photorealistic
rendering. However, both during training and inference, INR
methods require sampling a large number of rays and points in
three-dimensional space, with neural network inference at each
sample. This greatly increases computational cost and leads to
longer training times and slower inference speeds Tewari et al.
(2020, 2022). Although there are many works aimed at accel-
erating training and inference M¨uller et al. (2022); Fridovich-
Keil et al. (2023), it remains challenging to meet the practical
requirements of medical scenarios.
Recently, 3D Gaussian Splatting (3DGS) Kerbl et al. (2023)
has emerged as a promising technique for novel view synthe-
sis. It can achieve performance comparable to or better than
state-of-the-art INR methods, while reducing training time and
greatly improving rendering speed. Building on this progress,
many methods Yang et al. (2024d); Shan et al. (2025a); Li
et al. (2024a); Xie et al. (2024); Liu et al. (2024); Huang et al.
(2024b); Liu et al. (2025) have extended 3DGS to 4D surgi-
cal reconstruction. These methods model deformable scenes
using a single canonical space and a deformation network, sim-
ilar to INR approaches. The main difference is that they use
3DGS to represent the canonical space, leveraging its strengths
to improve both the quality and speed of deformable reconstruc-
tion. Despite these advances, several key challenges remain.
First, most existing methods Yang et al. (2024d); Shan et al.
(2025a); Li et al. (2024a); Xie et al. (2024); Liu et al. (2024);
Huang et al. (2024b); Liu et al. (2025) are designed for sce-
narios where the endoscope remains fixed (see Fig. 1(a)). In
these cases, a single canonical space and a single deformation
network can establish correspondences between the observed
space and the canonical space, effectively capturing scene de-
formations. However, when the endoscope moves significantly
(see Fig. 1(b) and Fig. 1(c)), new scene content constantly en-
ters the field of view, and the observed scene may be com-
pletely different from what is already represented in the canon-
ical space.
As a result, these methods struggle to associate
new observations with the canonical space. Second, to obtain
a good initialization, current state-of-the-art algorithms usually
rely on stereo depth priors or structure-from-motion (SfM) al-
gorithms such as COLMAP Schonberger and Frahm (2016) to
initialize the canonical space. However, for monocular endo-
scopes, depth priors often have scale ambiguity. Moreover, al-
though COLMAP performs well for static natural scenes, it has
difficulty with deformable endoscopic scenes, especially under
challenging conditions such as lighting changes and limited tex-
ture. Because of these limitations, current state-of-the-art meth-
ods often show a significant drop in reconstruction quality, or
even fail completely, when applied to monocular surgical recon-
struction with arbitrary camera motions (as shown in Fig. 1(d)).
To address these challenges, we propose Local-EndoGS, a
high-quality 4D surgical reconstruction framework for monoc-
ular endoscopic sequences with arbitrary camera motion. Sim-
ilar to state-of-the-art algorithms, our framework builds on 3D
Gaussian Splatting Kerbl et al. (2023) and leverages its efficient
rendering capabilities. To effectively capture dynamic changes
in long surgical sequences with significant camera motion, we
design a progressive, window-based global scene representa-
tion to model deformable surgical scenes. Specifically, we dy-
namically divide the input sequence into multiple local win-
dows based on the scene’s dynamics and use local deformable
scene representations to model the content observed in each
window. Each local deformable scene representation consists of
a local canonical space and a deformation field, with parameters
optimized progressively. This approach ensures scalability and
allows the framework to handle long sequences with significant
camera motion. To overcome unreliable initialization due to the
lack of stereo depth or accurate SfM in monocular endoscopic
sequences, we introduce a coarse-to-fine initialization strategy
for the local canonical space in each window. This strategy
integrates multi-view geometric information, cross-window in-
formation, and monocular depth priors to provide stable ini-
tialization and maintain scale consistency, eliminating the need
for stereo depth priors and COLMAP. Furthermore, we incor-
porate long-range 2D pixel trajectory constraints and physical
motion priors into our optimization framework to ensure that
the recovered deformations accurately reflect real tissue mo-
2

<!-- page 3 -->
tion. As shown in Fig. 1(d), experiments on public datasets
demonstrate that our method outperforms existing state-of-the-
art methods in reconstructing deformable scenes from monoc-
ular endoscopy under large camera motions.
In summary, our main contributions are as follows:
• A scalable 4D reconstruction framework for monoc-
ular endoscopy: To the best of our knowledge, Local-
EndoGS is the first framework that enables high-quality
4D reconstruction of deformable surgical scenes from
monocular endoscopic sequences with arbitrary camera
motion. Our method uses 3D Gaussian Splatting and a
progressive, window-based global scene representation to
efficiently and accurately model long sequences.
• Coarse-to-fine initialization strategy for monocular se-
quences: We introduce a robust coarse-to-fine initializa-
tion strategy for local canonical spaces. This strategy in-
tegrates multi-view geometry, cross-window information,
and monocular depth priors.
It removes the need for
stereo depth or accurate structure-from-motion, ensuring
stable and scale-consistent initialization for monocular se-
quences.
• Incorporation of long-range trajectory and motion pri-
ors in optimization: We incorporate long-range 2D pixel
trajectory constraints and physical motion priors into our
optimization framework. This enables more accurate and
robust deformation estimation when reconstructing dy-
namic surgical scenes from monocular sequences.
• Comprehensive evaluation and demonstration of effec-
tiveness: We conduct rigorous evaluations on multiple
datasets and perform thorough ablation studies to demon-
strate the effectiveness and advantages of Local-EndoGS.
2. Related work
In this work, we focus on 4D monocular surgical reconstruc-
tion under arbitrary camera motion. Thus, we primarily review
the following areas: 1) surgical reconstruction methods based
on non-implicit neural representations; 2) surgical reconstruc-
tion methods based on implicit neural representations; and 3)
techniques utilizing 3D Gaussian Splatting.
2.1. Surgical Reconstruction Based on Non-Implicit Neural
Representations
In recent decades, notable progress has been made in recon-
structing surgical scenes using non-implicit methods. These
techniques are widely used in clinical fields such as ortho-
pedics, otolaryngology, gastroenterology, and pulmonology
Schmidt et al. (2024); Cui et al. (2021); Xie et al. (2022). Early
methods often integrated Simultaneous Localization and Map-
ping (SLAM) with densification techniques to generate semi-
dense or dense representations of surgical scenes Mahmoud
et al. (2017, 2018); Chen et al. (2018); Marmol et al. (2019).
However, since these methods assume that the scene remains
static, their performance is limited when applied to deformable
environments. To overcome these limitations, researchers have
introduced algorithms tailored for non-rigid scenarios, such as
Non-Rigid Structure from Motion (NRSfM) Sengupta and Bar-
toli (2021); Lamarca et al. (2020) and Shape-from-Template
(SfT) approaches Cheema et al. (2018); Lamarca et al. (2020).
NRSfM methods frequently incorporate assumptions about tis-
sue motion, including low-rank shape models or isometric con-
straints, to make the problem more tractable. SfT approaches
first establish a scene template or utilize a predefined template
(for instance, sheet-like or tubular structures), and subsequently
align this template to each frame to monitor deformations. With
the development of deep learning, recent research has explored
learning-based stereo depth estimation and deformation track-
ing using sparse deformation fields Li et al. (2020); Long et al.
(2021). Although these methods enhance reconstruction flex-
ibility and accuracy, they continue to encounter difficulties in
managing topological variations and achieving photorealistic
rendering results.
2.2. Surgical Reconstruction Based on Implicit Neural Repre-
sentations
Implicit neural representations have emerged as a promising
technology in recent years, addressing the limitations of tradi-
tional surgical reconstruction algorithms through differentiable
rendering and neural networks Wang et al. (2022); Zha et al.
(2023); Yang et al. (2024a). EndoNeRF Wang et al. (2022)
was an early attempt to use implicit neural representations for
modeling deformable surgical scenes. It employs a canonical
neural radiance field and a time-varying deformation network
to simulate tissue deformation, achieving promising results.
EndoSurf Zha et al. (2023) builds on this approach by using
signed distance fields to represent scene geometry. Lerplane,
Forplane Yang et al. (2024a), and SDFPlane Li et al. (2024b)
improve training efficiency by decomposing four-dimensional
space into several orthogonal two-dimensional feature planes.
UW-DNeRF Shan et al. (2025b) further incorporates uncer-
tainty in depth priors and leverages local information. Despite
these advances, most methods assume a stationary endoscope
and typically rely on stereo depth priors. Several algorithms
are capable of handling scenarios involving endoscope mo-
tion. LightNeuS Batlle et al. (2023) reconstructs static scenes
from monocular endoscopic images while accounting for light-
ing changes.
ENeRF-SLAM Shan et al. (2024a) and DDS-
SLAM Shan et al. (2024b) use implicit neural representations
to build dense SLAM systems for endoscopy. UC-NeRF Guo
et al. (2024b) employs an uncertainty-aware conditional NeRF
for novel view synthesis in endoscopic scenes. However, most
of these methods are designed for static scenes and do not con-
sider scene deformation. In addition, they face challenges such
as long training times and low rendering efficiency due to the
computational complexity of implicit neural representations.
2.3. Surgical Reconstruction Based on 3D Gaussian Splatting
3D Gaussian Splatting (3DGS), introduced by Kerbl et al.
(2023), is an explicit radiance field method that enables effi-
cient and high-quality rendering of 3D scenes. To further im-
3

<!-- page 4 -->
Figure 2: Overview of Local-EndoGS. Given a long monocular endoscopic sequence with arbitrary camera motion, Local-EndoGS reconstructs the entire deformable
scene using a progressive window-based global scene representation (3.2). Specifically, the sequence is first divided into multiple local windows based on its dynamic
characteristics. For each local window, the scene structure is initialized using a local canonical space initialization strategy (3.4). Then, a local deformable scene
representation (3.3) is used to model each region. The parameters of each local scene representation are then optimized using carefully designed loss functions (3.5)
in a progressive manner until all local models are fully optimized.
prove training and rendering efficiency, many approaches uti-
lize 3DGS in the canonical space and combine it with a defor-
mation field to model deformable surgical scenes. The defor-
mation field can be represented in various ways. For example,
Deform3DGS Yang et al. (2024d) and EH-SurGS Shan et al.
(2025a) use explicit basis functions, while EndoSparse Li et al.
(2024a) and SurgicalGaussian Xie et al. (2024) employ mul-
tilayer perceptrons (MLPs) to capture deformation. LGS Liu
et al. (2024), Endo-4DGS Huang et al. (2024b), and Endo-
Gaussian Liu et al. (2025) combine multiple orthogonal 2D
feature planes with a compact MLP. Compared to implicit neu-
ral representation methods, these approaches significantly im-
prove training speed and rendering efficiency. However, they
share similar limitations with implicit methods: they assume
a stationary endoscope and typically rely on stereo depth pri-
ors or precise structure-from-motion for initialization and op-
timization, which limits their applicability in real surgical sce-
narios. Free-SurGS Guo et al. (2024a), EndoGSLAM Wang
et al. (2024a) and Endoflow-SLAM Wu et al. (2025) use 3DGS
for surgical scene reconstruction and joint camera pose op-
timization.
Endo-2DTAM Huang et al. (2025) introduces a
surface-normal-aware pipeline based on 2D Gaussian distribu-
tions Huang et al. (2024a). Gaussian Pancakes Bonilla et al.
(2024) combined 3D Gaussian Splatting with RNN-SLAM Ma
et al. (2021) to enable real-time, high-quality 3D reconstruc-
tion of endoscopic scenes, offering substantial gains in both ren-
dering accuracy and computational efficiency. However, these
methods are designed for static scenes and do not address the
deformable nature of endoscopic images. In this work, we ad-
dress these limitations by proposing a 4D reconstruction algo-
rithm that supports arbitrary camera motion and enables recon-
struction from monocular endoscopic images, providing high-
quality reconstruction of deformable surgical scenes.
4

<!-- page 5 -->
3. Methodology
Given a sequence of monocular endoscopic images {Ii}Q
i=1
with arbitrary but known poses {Ei = (Ri, ti)}Q
i=1 and intrinsic
parameters K, our goal is to develop a 4D monocular surgical
reconstruction framework based on 3DGS that accurately cap-
tures deformable scenes with fine detail and geometric struc-
ture. The Local-EndoGS pipeline is illustrated in Fig. 2 and
consists of four main components: 1) A progressive window-
based global scene representation that adaptively divides the
input sequence into smaller windows and progressively opti-
mizes the parameters of each local model to represent the scene
within each window (Sec. 3.2); 2) A local deformable scene
representation that models the deformable scene within each
window (Sec. 3.3); 3) A local canonical space initialization
that reliably initializes the local canonical space in each window
to ensure stable model performance (Sec. 3.4); 4) A carefully
designed loss function that optimizes the model by incorporat-
ing monocular images, long-range 2D pixel trajectory priors,
and physical motion priors (Sec. 3.5).
3.1. Preliminaries: 3DGS-based Deformable Surgical Recon-
struction
3D Gaussian Splatting. 3DGS Kerbl et al. (2023) represents
3D scenes explicitly using anisotropic 3D Gaussian functions
ϕ = {gi(x)}. Each Gaussian gi is defined by its position µ ∈
R3, covariance matrix Σ ∈R3×3, opacity σ ∈R, and view-
dependent color c ∈R3, which is parameterized by spherical
harmonics Fridovich-Keil et al. (2022). The distribution of a
3D Gaussian is given by:
gi(x) = e−1
2 (x−µ)T Σ−1(x−µ)
(1)
To ensure that Σ remains a valid covariance matrix during op-
timization, it is parameterized as Σ = RS S TRT, where S is a
diagonal scale matrix and R is a rotation matrix, defined by a
scaling vector s ∈R3 and a quaternion r ∈R4.
To render an image from a given viewpoint, the 3D Gaus-
sians are projected onto the 2D image plane using splatting
techniques Zwicker et al. (2002). The corresponding 2D co-
variance matrix Σ′ and center µ′ in camera coordinates are com-
puted as Σ′ = JWΣWT JT and µ′ = JWµ, where W denotes the
viewing transformation and J is the Jacobian of the affine ap-
proximation of the projective transformation. For each pixel p,
the color ˆC(p) is computed by alpha blending:
ˆC(p) =
X
i∈N
ciαi
i−1
Y
j=1

1 −α j

(2)
Here, N is the set of all Gaussians that influence pixel p along
the viewing ray, and the final opacity αi is given by the product
of the learned opacity σi and the Gaussian:
αi = σi exp
 
−1
2(p −µ′
i)TΣ′−1(p −µ′
i)
!
(3)
where µ′
i is the projected 2D center of the i-th Gaussian in the
camera coordinate system. Similarly, the depth ˆD(p) is ren-
dered as:
ˆD(p) =
X
i∈N
diαi
i−1
Y
j=1

1 −α j

(4)
Here, di is the z-depth coordinate of the i-th 3D Gaussian in
view space.
3DGS-based Deformable Surgical Reconstruction.
Re-
cent algorithms for deformable surgical scene reconstruc-
tion Yang et al. (2024d); Shan et al. (2025a); Li et al. (2024a);
Xie et al. (2024); Liu et al. (2024); Huang et al. (2024b); Liu
et al. (2025) typically use a single canonical space, defined at a
reference time (often t = 0), and represented by 3D Gaussians.
To model tissue motion and deformation during surgery, a de-
formation network Dt predicts changes in the center position,
rotation, and scale for each Gaussian at each time step t:
(∆µt, ∆rt, ∆st) = Dt(µ0, r0, s0, t)
(5)
Here, µ0, r0, s0 are the center coordinates, rotation quaternion,
and scaling factors in the canonical space. ∆µt, ∆rt, ∆st are the
temporal offsets at time t relative to their canonical values. The
parameters at time t are obtained by adding these offsets to the
canonical values:
(µt, rt, st) = (µ0 + ∆µt, r0 + ∆rt, s0 + ∆st)
(6)
Finally, during training, RGB and depth images are rendered
using (2) and (4), respectively. The ground truth RGB images
are provided as inputs, while depth priors are obtained from a
stereo depth estimation network Li et al. (2021). The RGB and
depth losses are computed by comparing the rendered images
with the ground truth RGB images and estimated depth priors,
respectively. The parameters of both the deformation network
and the 3D Gaussians are jointly optimized by minimizing the
combined loss.
3.2. Progressive Window-based Global Scene Representation
Modeling a long sequence with a single deformable 3DGS
representation is challenging, especially when significant cam-
era motion occurs. Large camera movements disrupt the cor-
respondence between the observed and canonical spaces, often
resulting in reconstruction failure. To address this limitation,
we propose a progressive, window-based global scene repre-
sentation.
As shown in Fig. 2, our method first uses adaptive window
partitioning to divide the input sequence into M contiguous
local windows.
Each window is modeled using a local de-
formable scene representation (see Sec. 3.3). The entire scene
is then reconstructed from multiple local models, denoted as
{ϕi, Dt
i}M
i=1. During training, the parameters of each local model
are optimized progressively.
Specifically, the frames within
window i −1 are used to optimize the parameters of the corre-
sponding local model {ϕi−1, Dt
i−1}. After optimizing {ϕi−1, Dt
i−1},
we save its parameters and move on to optimize those for win-
dow i. This process is repeated until all M local models have
been optimized. The progressive optimization strategy enables
5

<!-- page 6 -->
our method to efficiently handle sequences of arbitrary length
and various types of camera motion.
To balance model accuracy and training efficiency, we pro-
pose an adaptive method that determines window sizes based
on the dynamic characteristics of the input sequence. We an-
alyze sequence dynamics from two perspectives: camera mo-
tion and frame content variation. For camera motion, we mea-
sure both translational and rotational pose changes through-
out the ground-truth camera poses of each sequence. A new
window is created when either the translational difference ex-
ceeds the threshold δt or the rotational angle difference exceeds
the threshold δr. This approach ensures consistent viewpoints
within each window and reduces representation errors caused
by large viewpoint changes. For frame content variation, pre-
vious methods Shaw et al. (2024) rely on optical flow to esti-
mate scene changes. However, optical flow is often unreliable
in endoscopic environments due to illumination changes and
textureless tissue surfaces. Instead, we use a simple yet effec-
tive method by comparing RGB differences between each frame
and the first frame of the current window. A new window is cre-
ated when this difference exceeds a predefined threshold (em-
pirically set to 0.05). Compared to using a fixed window size as
a hyperparameter, our adaptive partitioning strategy increases
the sampling frequency in highly dynamic regions and reduces
it in low-motion areas. This approach leads to improved system
performance, as demonstrated in the experimental results (see
Sec. 5.5).
3.3. Local Deformable Scene Representation
Dividing the input sequence into multiple local windows
with similar content (as described in Sec. 3.2) ensures that
the content within each window can be effectively modeled
using a single canonical space and a single deformation net-
work. Therefore, any 3DGS-based deformable reconstruction
method can be applied to each window. In this work, we use
EH-SurGS Shan et al. (2025a) for this purpose.
Specifically, for each local window, we construct a canonical
space using 3D Gaussians (initialized as described in Sec. 3.4)
and model local deformations using a deformation network.
The deformation network predicts the temporal evolution of
both the spatial parameters and the opacity of each Gaussian,
which are defined as learnable functions of time. This design al-
lows the model to flexibly capture both general and irreversible
deformations:
xt = x0 +
B
X
j=1
ωx
jbx(t).
(7)
Here, xt denotes the mean, rotation, scale, or opacity; x0 is the
canonical value; b(t) is a Gaussian basis function with learnable
center and variance; and ω j are learnable weights.
To handle irreversible and dynamic scene changes caused by
intraoperative operations such as tissue shearing, EH-SurGS in-
troduces a life-cycle mechanism for 3D Gaussians. Each Gaus-
sian is activated only within its valid temporal range and deacti-
vated once the corresponding structure disappears. To improve
computational efficiency, EH-SurGS uses an adaptive motion
hierarchy strategy to distinguish between deformable and static
Figure 3: Comparison of Feature Matching and Point Cloud Results Using Tra-
ditional Methods and Track-Any-Point (TAP) model Chen et al. (2024). (a)
Correspondences obtained using SIFT keypoints with brute-force matching. (b)
Sparse point cloud reconstructed from the correspondences shown in (a). (c)
Correspondences from the TAP model. (d) Dense point cloud from TAP-based
correspondences. Green and red lines indicate correct and incorrect feature
matches, respectively.
regions in each local scene. A dynamically updated mask sep-
arates these regions based on average deformation and the con-
sistency of rendering loss. This approach allows for efficient
resource allocation during training and inference. More details
are provided in our previous work Shan et al. (2025a).
3.4. Local Canonical Space Initialization
A well-initialized 3D Gaussian representation in the canoni-
cal space is important for effective model optimization and per-
formance. Previous methods often use stereo depth priors or
structure-from-motion (SfM) point clouds for scene initializa-
tion Yang et al. (2024d); Shan et al. (2025a); Li et al. (2024a);
Xie et al. (2024); Liu et al. (2024); Huang et al. (2024b); Liu
et al. (2025). However, for monocular sequences, scale ambi-
guity in depth estimation makes initialization challenging. In
addition, unique characteristics of endoscopic scenes often re-
sult in sparse and unstable SfM point clouds. To address these
issues, we propose a coarse-to-fine initialization method for
monocular sequences. Our approach combines multi-view ge-
ometry, cross-window information, and monocular depth priors
to achieve stable and consistent initialization.
3.4.1. Coarse Stage: Scale-Aware Initialization
In this stage, we construct a dense point cloud to initialize
3D Gaussians in the local canonical space. This point cloud
captures the basic geometry and maintains a consistent global
scale, providing a stable basis for the next stage.
For each local window i, composed of an image set S i =
{Ii
j}mi
j=0 with timestamps {ti
j}mi
j=0, we use the known ground-
truth poses of all frames to generate a dense triangulated point
cloud. A common traditional approach is to use traditional fea-
ture extraction and matching methods, such as SIFT keypoints
Lowe (2004) with brute-force matching in OpenCV, followed
by multi-view triangulation to obtain an initial point cloud.
However, as shown in Fig. 3a, challenges such as illumination
changes, repetitive low-texture patterns, and tissue deformation
in endoscopic scenes can lead to incorrect feature matching and
6

<!-- page 7 -->
Figure 4: Visualization of RGB images rendered by 3DGS ϕC
i initialized from
the coarse stage and their corresponding reconstruction error maps with respect
to the ground truth. Top row: rendered RGB images. Bottom row: pixel-wise
reconstruction error maps, where higher values indicate greater reconstruction
errors, especially near tissue boundaries and regions with deformation.
sparse feature points. As a result, the generated point cloud is
often sparse (Fig. 3b). To address this, we leverage the Track-
Any-Point (TAP) model Chen et al. (2024), a 2D vision foun-
dation model built upon the CoTracker framework Karaev et al.
(2024), to perform end-to-end point tracking for establishing
correspondences across multiple image frames. Specifically,
we use this pre-trained TAP model to extract pixel-wise tempo-
ral trajectories for K sampled pixels. These trajectories are de-
noted as Ti = {U(pq)(ti
j) | q = 1, . . . , K; j = 0, . . . , mi}, where
pq ∈R2 is the location of the q-th pixel in Ii
0, and U(pq)(ti
j) is
its 2D position in the j-th frame of window i. We then select k
image pairs from window i and use Ti to find correspondences
across frames, as shown in Fig. 3b. These correspondences are
triangulated using known camera intrinsics and poses to recover
3D points. The point clouds from multiple pairs are merged to
obtain a scale-consistent dense point cloud Ptri
i . As shown in
Fig. 3d, the point cloud obtained using our TAP-based method
is much denser than that produced by traditional methods and
better represents the geometric structure of the scene.
This
point cloud is used to initialize the 3D Gaussians ϕtri
i , follow-
ing Kerbl et al. (2023), and serves as a global scale reference
for later stages.
In addition, we introduce a cross-window information prop-
agation strategy. With our progressive window-based global
scene representation, we can efficiently propagate and integrate
information across adjacent windows. The core idea is to use
the optimized local deformable scene representation from win-
dow i −1 to estimate the initial canonical space in window i,
transferring this prior knowledge forward. Specifically, the op-
timized local window i −1 is represented by ϕi−1 and its defor-
mation network Dt
i−1. We apply Dt
i−1 to the 3D Gaussians in
ϕi−1 to predict their parameters at the observation time ti
0 in the
canonical space of window i, as shown in Eq. (5) and Eq. (6).
This process produces a set of 3D Gaussians for window i, de-
noted as ϕwin
i
. We then fuse ϕwin
i
with the triangulated repre-
sentation ϕtri
i to form the initial representation ϕC
i for the coarse
stage. For the first window, we use only the triangulated point
cloud for coarse stage initialization.
3.4.2. Fine Stage: Error-Guided Region Refinement
As shown in Fig. 4, we observe noticeable reconstruction er-
rors in some regions, especially near tissue boundaries, specular
reflections, and areas with significant deformation. To address
this, we propose an error-guided region refinement strategy that
incorporates monocular depth priors. Specifically, we first ren-
der the RGB image and the corresponding depth map for the
first frame Ii
0 of window i using the current initialization ϕC
i ,
denoted as Irender and Drender, according to Eq. (2) and Eq. (4),
respectively. At the same time, we use a pretrained monocular
depth estimation network Yang et al. (2024c) to predict a depth
map Dest from Ii
0. We find that Drender benefits from strict ge-
ometric consistency with the reconstructed 3D scene, but may
be inaccurate or incomplete in regions affected by erroneous
3D initialization. In contrast, Dest can provide consistent depth
estimates even in challenging regions, but suffers from scale
ambiguity. Therefore, we align the monocular depth with the
rendered depth by estimating a scale factor α and an offset β
using least-squares fitting:
Dfine = α · Dest + β.
(8)
where
α, β = arg min
α,β ∥Drender −(αDest + β)∥2
2.
(9)
Dfine combines the strengths of both sources and mitigates their
individual limitations.
Next, we compute a per-pixel photometric error map by com-
paring Irender with the observed image Ii
0. For regions with low
error (empirically set to 0.7), we retain the initialization from
the coarse stage. For regions with high error, we use the aligned
depth map Dfine to back-project the corresponding pixels into
3D space. We then generate new 3D points that are fused with
the original 3D Gaussians to refine the geometry. This selective
refinement is both computationally efficient and helps preserve
well-initialized geometry in areas with low error.
In summary, our initialization strategy first uses multi-view
geometry and cross-window information to address scale ambi-
guity in monocular depth estimation. In cases where the cam-
era remains stationary and multi-view geometry becomes inef-
fective, we utilize cross-window information to maintain scale
consistency among local windows. Monocular depth priors are
then applied in the fine stage to refine regions where the coarse
initialization is less accurate. This approach is specifically de-
signed for monocular sequences and does not require stereo
depth information. As demonstrated by our experimental re-
sults (see Sec. 5.5), this improves the robustness and accuracy
of the overall initialization.
3.5. Optimization
We carefully design three loss functions to optimize the lo-
cal deformable representation in posed monocular endoscopic
sequences.
Rendering Loss. The rendering loss enforces color consis-
tency between the rendered and observed images for each frame
within a window. During training, we render images ˆI using (2)
7

<!-- page 8 -->
Table 1: Summary of the datasets and sequences used in our study. Each sequence’s number of frames, image resolution, camera motion type, and window count
are listed.
Dataset
Sequence
Frames
Resolution
Camera Motion
Windows
EndoNeRF Wang et al. (2022)
Pulling
63
512 × 640
Fixed camera
1
Cutting
156
512 × 640
Fixed camera
1
StereMIS Hayoz et al. (2023)
Sequence1
1000
512 × 640
Moving around tissue
49
Sequence2
1500
512 × 640
Moving around tissue
69
EndoMapper Azagra et al. (2023)
Sequence1
267
512 × 640
Moving forward
15
Sequence2
267
512 × 640
Moving forward
16
Sequence3
267
512 × 640
Moving forward
20
and compute the rendering loss Lrgb as:
Lrgb = (1 −M) ((1 −λ)L1 + λLD−S S IM) .
(10)
Here, M is a mask that excludes regions containing surgical
instruments, as in previous work Yang et al. (2024d); Shan
et al. (2025a); Li et al. (2024a); Xie et al. (2024); Liu et al.
(2024); Huang et al. (2024b); Liu et al. (2025). The weight
λ is empirically set to 0.2. L1 is the pixel-wise L1 loss be-
tween the rendered image ˆI and the input image I, i.e., L1 =
∥ˆI −I∥1.
LD−S S IM is the structural dissimilarity loss Kerbl
et al. (2023). This objective balances pixel-level accuracy and
perceptual similarity, following the approach in 3D Gaussian
Splatting Kerbl et al. (2023). Notably, we do not use depth
priors as a supervision signal, as we found that this negatively
impacts the performance of our method (see Sec. 5.6 for de-
tails).
2D Tracking Loss. Leveraging the stable correspondences
across multiple frames provided by the Track-Any-Point (TAP)
model Chen et al. (2024), we construct a 2D tracking loss. This
loss enforces consistency between the canonical and observed
spaces by supervising pixel-wise temporal trajectories in the
RGB frames. Following Wang et al. (2024b), for each local
window i, we first rasterize the motion of 3D Gaussians in the
canonical space at the observation time ti
0 into the query frame
Ii
j at time ti
j. Specifically, we compute a 3D trajectory map
ˆXw
ti
0→ti
j ∈RH×W×3, which provides the 3D world coordinates at
time ti
j corresponding to each 2D pixel location initially ob-
served at ti
0:
ˆXw
ti
0→ti
j(p) =
X
i∈H(p)
µi,ti
0→ti
jαi
i−1
Y
j=1

1 −α j

,
(11)
where H(p) is the set of Gaussians intersecting pixel p at ti
0, and
µi,ti
0→ti
j is the center of Gaussian i at ti
j, computed from (7). We
then project these 3D points to the image plane using camera
intrinsics K and extrinsics Et:
ˆUti
0→ti
j(p) = Π

KEt ˆXw
ti
0→ti
j(p)

,
(12)
where Π(·) denotes the standard perspective projection. This
process establishes pixel-level correspondences across frames.
We supervise the rendered trajectories using 2D pixel tracks
from the TAP model, as described in Sec. 3.4:
Ltrack =
Uti
0→ti
j −ˆUti
0→ti
j
1 .
(13)
Physics-Based Regularization.
To improve the physical
plausibility and motion consistency of 3D Gaussian deforma-
tion modeling in local deformable scene representation, we in-
troduce three physics-based spatial constraints to regularize the
transformation of Gaussians from the canonical space to the ob-
servation space: short-term local rigidity loss (Lrigid), local ro-
tation similarity loss (Lrot), and long-term local isometry loss
(Liso). These losses are computed for each Gaussian and its k
nearest neighbors (kNN) as follows Luiten et al. (2024):
Lx = 1
|G|
X
i∈G
X
j∈kNN(i)
wi, jLx,i, j,
(14)
where x ∈{rigid, rot, iso}, |G| is the number of Gaussians in
the canonical space, and wi, j is an isotropic Gaussian weight
that reflects the spatial relationship between the i-th and j-th
Gaussians:
wi, j = exp

−λw
µj,c −µi,c
2
2

.
(15)
Here, λw = 2000 defines the standard deviation. µi,c and µ j,c de-
note the positions of the i-th and j-th Gaussians in the canonical
space c, respectively.
The rigidity loss, Lrigid, encourages adjacent Gaussians
within a local region to undergo similar rigid transformations.
This helps preserve local structure and suppress unnatural de-
formations:
Lrigid,i, j =
(µ j,τ −µi,τ) −∆Ri(µj,c −µi,c)
2 .
(16)
Here, ∆Ri = Ri,τR−1
i,c is the relative rotation of the i-th Gaussian
between the canonical space c and the observation space τ.
The rotation similarity loss, Lrot, promotes consistent rota-
tion among neighboring Gaussians and reduces abrupt angular
changes in local regions:
Lrot,i, j =
ˆq j,τ ˆq−1
j,c −ˆqi,τ ˆq−1
i,c
2
2 .
(17)
Here, ˆq is the unit quaternion representing the rotation of each
Gaussian.
8

<!-- page 9 -->
Finally, the isometry loss, Liso, preserves the relative dis-
tances between neighboring Gaussian centers over time:
Liso,i, j =
µj,c −µi,c
2 −
µj,τ −µi,τ
2 .
(18)
Training Loss. The total training loss is defined as:
L = λrgbLrgb + λtrackLtrack + λrigidLrigid + λrotLrot + λisoLiso.
(19)
where λrgb, λtrack, λrigid, λrot, and λiso are the weights for each
loss term. In our experiments, we set λrgb = 1, λtrack = 0.01,
λrigid = 0.05, λrot = 0.05, and λiso = 0.05.
4. Experiment Setup
4.1. Datasets
We evaluate Local-EndoGS on three endoscopic datasets
that feature deformable scenes and varying camera motions:
the EndoNeRF dataset Wang et al. (2022), the StereoMIS
dataset Hayoz et al. (2023), and the EndoMapper dataset Aza-
gra et al. (2023). Table 1 provides a detailed summary of the
dataset sequences.
EndoNeRF dataset.
The EndoNeRF dataset Wang et al.
(2022) contains six video clips recorded during a Da Vinci
robotic prostatectomy procedure, with the camera remaining
stationary (see Fig. 1(a)); therefore, its extrinsic parameters are
fixed to the identity matrix (R = I, t = 0). Each clip has a
resolution of 512×640 and a duration of 4–8 seconds at 15 fps.
We use two public sequences, Pulling and Cutting, which de-
pict non-rigid soft tissue deformations and contain 63 and 156
frames, respectively. We follow the protocol outlined in En-
doNeRF Wang et al. (2022) for handling surgical tool occlu-
sions. Only left-view images are used for training and evalua-
tion.
StereoMIS dataset.
The StereoMIS dataset Hayoz et al.
(2023) is an in vivo collection recorded with the da Vinci Xi
surgical robot. Ground-truth camera poses are obtained from
the endoscope’s forward kinematics and synchronized with the
video streams. This dataset includes 16 videos captured from
three pigs (P1, P2, and P3) and three human subjects (H1, H2,
and H3), with tissue deformations caused by breathing and ma-
nipulation. For our experiments, we select two sequences con-
taining 1,000 and 1,500 consecutive frames, each with a reso-
lution of 512×640. The camera moves around the deformable
tissue (see Fig. 1(b)). We use the provided surgical tool masks
to address occlusions. Only left-view images are used for train-
ing and evaluation.
EndoMapper dataset.
The EndoMapper dataset Azagra
et al. (2023) is an open-source collection widely used in med-
ical SLAM research, comprising 59 video sequences. In our
experiments, we use the colon deformation subset generated
with the VR-Caps simulator ˙Incetan et al. (2021). This sub-
set simulates a forward colonoscope insertion procedure (see
Fig. 1(c)), where colon deformation is modeled using a sine
wave: Vt
y = V0
y + A sin(ωt + V0
x + V0
y + V0
z ), where V0
x, V0
y ,
and V0
z are the coordinates of surface points at rest. The am-
plitude A and frequency ω control the extent of deformation.
Table 2: Depth error and accuracy metrics used for evaluation. Here, d and d∗
denote the predicted and ground-truth depth values, respectively, and D repre-
sents the set of predicted depth values.
Metric
Definition
Abs Rel
1
|D|
P
d∈D
|d−d∗|
d∗
Sq Rel
1
|D|
P
d∈D
(d−d∗)2
d∗
RMSE
q
1
|D|
P
d∈D(d −d∗)2
RMSE log
q
1
|D|
P
d∈D(log d −log d∗)2
Accuracy (δ < 1.25)
1
|D|
P
d∈D I

max
 d
d∗, d∗
d

< 1.25

Accuracy (δ < 1.252)
1
|D|
P
d∈D I

max
 d
d∗, d∗
d

< 1.252
Each sequence also provides ground-truth camera poses that
are directly obtained from the simulator. We use three syn-
thetic sequences, Sequence1, Sequence3, and Sequence5. For
convenience, these sequences are referred to as Sequence1, Se-
quence2, and Sequence3, respectively, in the following exper-
iments. After removing images with camera poses that do not
clearly correspond to the visual content, each sequence contains
267 images at a resolution of 512×640.
4.2. Baseline methods
To evaluate the performance of our method, we compare it
with state-of-the-art approaches for deformable scene recon-
struction in endoscopic environments. These include EndoN-
eRF Wang et al. (2022), EndoSurf Zha et al. (2023), For-
plane Yang et al. (2024b), Endo-GS Zhu et al. (2024), De-
form3DGS Yang et al. (2024d), SurgicalGaussian Xie et al.
(2024), LGS Liu et al. (2024), EndoGaussian Liu et al. (2025),
EH-SurGS Shan et al. (2025a), and DDS-SLAM Shan et al.
(2024b). Among these, EndoNeRF, EndoSurf, and Forplane
use implicit neural representations. Endo-GS, Deform3DGS,
SurgicalGaussian, LGS, EndoGaussian, and EH-SurGS are
based on 3DGS for deformable surgical scene reconstruction.
Most of these methods are originally developed for the EndoN-
eRF and StereoMIS datasets rather than for EndoMapper. To
ensure a fair and comprehensive comparison, we additionally
include three baselines on the EndoMapper dataset: ENeRF-
SLAM Shan et al. (2024a), EndoGSLAM Wang et al. (2024a),
and Endo-2DTAM Huang et al. (2025), which represent the
current state-of-the-art approaches for colonoscopy video re-
construction. For all SLAM-based methods, including DDS-
SLAM, ENeRF-SLAM, EndoGSLAM, and Endo-2DTAM, we
disable the tracking thread and provide the ground-truth camera
poses as input, focusing solely on reconstruction performance.
All baseline methods are reproduced using their official reposi-
tories with the same RGB and depth inputs.
4.3. Evaluation metrics
Following previous work Zha et al. (2023); Shan et al.
(2025a); Guo et al. (2024b), we report quantitative comparisons
for both appearance and geometry.
For appearance quality,
we use standard metrics: Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index (SSIM) Wang et al. (2004), and
9

<!-- page 10 -->
Table 3: Quantitative comparison of different methods on the EndoNeRF dataset. Columns highlighted in
indicate that higher values are better, while those in
indicate that lower values are better. The best results are highlighted in bold, and the second best results are underlined.
Method
PSNR
SSIM
LPIPS
Abs Rel
Sq Rel
RMSE
RMSE log
δ < 1.25
δ < 1.252
Pulling
EndoNeRF
37.723
0.949
0.088
0.354
8.500
22.607
0.390
0.421
0.746
EndoSurf
37.117
0.950
0.108
9.854
943.824
39.685
1.049
0.325
0.583
ForPlane
30.555
0.896
0.104
0.313
8.802
26.267
0.344
0.425
0.787
DDS-SLAM
25.508
0.797
0.371
0.431
9.363
18.062
0.448
0.425
0.671
EndoGS
25.663
0.852
0.355
0.126
4.272
15.771
0.176
0.807
0.927
SurgicalGaussian
25.805
0.853
0.368
0.177
3.557
18.457
0.185
0.837
0.908
LGS
26.580
0.907
0.328
0.251
4.480
16.237
0.189
0.828
0.911
EndoGaussian
37.190
0.955
0.066
0.171
2.568
12.012
0.195
0.816
0.936
Deform3DGS
37.987
0.959
0.070
0.219
2.713
12.339
0.183
0.861
0.917
EH-SurGS
38.433
0.961
0.064
0.212
2.658
12.311
0.181
0.871
0.925
Local-EndoGS
38.727
0.964
0.053
0.119
1.648
6.933
0.147
0.915
0.988
Cutting
EndoNeRF
35.962
0.936
0.094
0.323
8.836
24.244
0.348
0.410
0.803
EndoSurf
34.942
0.938
0.119
3.654
267.044
31.563
0.799
0.389
0.585
ForPlane
25.951
0.833
0.122
0.503
14.695
22.967
0.421
0.427
0.747
DDS-SLAM
26.415
0.783
0.382
0.366
6.050
16.182
0.417
0.324
0.617
EndoGS
24.257
0.820
0.378
0.296
3.686
14.587
0.274
0.667
0.843
SurgicalGaussian
25.077
0.837
0.354
0.175
2.347
10.536
0.163
0.823
0.942
LGS
25.120
0.894
0.342
0.357
4.565
17.279
0.371
0.562
0.685
EndoGaussian
38.040
0.960
0.052
0.332
4.121
15.815
0.291
0.642
0.778
Deform3DGS
37.923
0.961
0.054
0.236
2.812
12.475
0.200
0.729
0.894
EH-SurGS
39.457
0.967
0.039
0.233
2.748
12.107
0.196
0.738
0.908
Local-EndoGS
39.647
0.968
0.037
0.107
1.136
5.825
0.135
0.905
0.973
Learned Perceptual Image Patch Similarity (LPIPS) Zhang
et al. (2018). For geometry, we assess depth prediction accuracy
using common metrics from prior studies Shao et al. (2022);
Guo et al. (2024b): Absolute Relative Error (Abs Rel), Squared
Relative Error (Sq Rel), Root Mean Square Error (RMSE),
RMSE log, and accuracy under the threshold δ < t, where
t ∈{1.25, 1.252}. The definitions of all error and accuracy met-
rics are summarized in Table 2. Following standard procedure
for monocular depth estimation, we apply median scaling dur-
ing evaluation. The scaling factor is computed as the ratio of
the medians of the ground-truth and predicted depth maps.
4.4. Implementation Details
We implement our method in PyTorch. All experiments are
conducted on an Ubuntu 20.04 system with a single RTX 4090
GPU and an Intel Xeon Platinum 8474 CPU. We use the Adam
optimizer with an initial learning rate of 1.6 × 10−3, and other
training parameters are set as in the original 3DGS. Each lo-
cal window is trained for 1,000 iterations. Following previ-
ous work Zha et al. (2023); Shan et al. (2025b), we split each
dataset into training and testing sets using a 7:1 ratio. We quan-
tify translational difference using the mean squared difference
(MSE) between translation vectors.
For the StereoMIS, the
translational threshold is set to δt = 0.6 cm2, corresponding
to an approximate physical displacement of 7.7 cm. For the
EndoMapper, δt = 0.6 mm2, corresponding to about 7.7 mm of
displacement. For both datasets, the rotational threshold is fixed
at δr = 15◦. The number of windows for each sequence, deter-
mined by our adaptive window partitioning method, is summa-
rized in Table 1. To ensure fairness, we run all methods on each
dataset three times and report the average results.
5. Experimental results
Quantitative results on the three datasets are summarized in
Table 3, Table 4, and Table 5. Our method consistently achieves
higher appearance rendering quality and more accurate depth
prediction than existing approaches across all datasets. Quali-
tative results, shown in Fig. 5 and Fig. 6, further confirm these
findings by demonstrating that our approach produces visu-
ally realistic renderings and accurate depth maps under various
camera motions. Overall, these results demonstrate the effec-
tiveness and robustness of our method for 4D monocular surgi-
cal reconstruction.
5.1. Results on the EndoNeRF Dataset
As shown in Table 3, on the EndoNeRF dataset with a fixed
camera view, Local-EndoGS achieved the best performance on
both sequences. In the Pulling and Cutting sequences, Local-
EndoGS obtained the highest scores for image quality metrics
such as PSNR and SSIM (38.727/0.964 and 39.647/0.968, re-
spectively), and the lowest LPIPS values (0.053 and 0.037),
indicating better image reconstruction quality. For geometric
reconstruction, baseline methods rely heavily on stereo depth
10

<!-- page 11 -->
Table 4: Quantitative comparison of different methods on the StereoMIS dataset. Columns highlighted in
indicate that higher values are better, while those
highlighted in
indicate that lower values are better. The best results are shown in bold, and the second-best results are underlined.
Method
PSNR
SSIM
LPIPS
Abs Rel
Sq Rel
RMSE
RMSE log
δ < 1.25
δ < 1.252
StereoMIS-Sequence1
EndoNeRF
25.913
0.746
0.481
0.420
23.226
58.555
0.478
0.310
0.590
EndoSurf
24.883
0.715
0.520
3.219
472.041
51.691
0.736
0.319
0.578
ForPlane
16.389
0.651
0.792
0.640
36.502
62.023
0.613
0.299
0.544
DDS-SLAM
16.585
0.569
0.673
0.355
20.031
40.680
0.402
0.462
0.726
EndoGS
20.412
0.691
0.643
0.572
10.705
29.252
0.522
0.454
0.740
SurgicalGaussian
21.149
0.714
0.558
0.678
12.574
32.563
0.596
0.396
0.611
LGS
18.540
0.560
0.620
0.753
15.270
34.863
0.718
0.326
0.524
EndoGaussian
20.440
0.690
0.650
0.618
11.384
31.157
0.553
0.453
0.669
Deform3DGS
20.497
0.697
0.646
0.582
10.928
27.536
0.541
0.513
0.707
EH-SurGS
21.161
0.716
0.556
0.574
10.754
27.419
0.529
0.531
0.724
Local-EndoGS
32.294
0.892
0.304
0.112
2.951
7.485
0.149
0.919
0.976
StereoMIS-Sequence2
EndoNeRF
11.857
0.433
0.773
1.632
123.563
81.365
0.901
0.157
0.313
EndoSurf
23.614
0.601
0.677
0.831
52.405
33.187
0.593
0.359
0.622
ForPlane
23.123
0.589
0.586
1.978
86.499
60.861
0.819
0.209
0.436
DDS-SLAM
19.577
0.523
0.662
0.280
11.284
29.105
0.333
0.557
0.828
EndoGS
15.493
0.635
0.692
1.254
55.003
42.173
0.797
0.277
0.504
SurgicalGaussian
20.940
0.717
0.603
1.807
59.938
45.967
0.869
0.197
0.392
LGS
22.393
0.733
0.586
2.213
78.488
57.582
1.130
0.105
0.218
EndoGaussian
23.927
0.753
0.575
1.774
58.630
42.205
0.826
0.242
0.471
Deform3DGS
21.453
0.719
0.598
1.217
43.830
35.641
0.728
0.293
0.551
EH-SurGS
22.560
0.741
0.587
1.201
43.173
35.174
0.719
0.298
0.569
Local-EndoGS
31.487
0.922
0.297
0.129
3.852
9.271
0.154
0.912
0.967
Figure 5: Qualitative comparison of image rendering and depth prediction on deformable scenes at different time points from the StereoMIS dataset. Each pair of
rows represents a specific time point: the first row shows the rendered RGB images, and the second row shows the predicted depth maps. The figure presents results
from three time points (from top to bottom), illustrating how the observed scene changes as the camera moves. Compared to existing methods, Local-EndoGS
(Ours) consistently provides finer reconstruction details, while the baseline methods show limitations in both image quality and depth accuracy.
as an additional input. When only monocular sequences are
available, the scale ambiguity of monocular depth priors leads
to significant errors in geometric reconstruction. Our method
addresses this problem and substantially outperforms baseline
methods in depth prediction. For example, on the Pulling se-
quence, Local-EndoGS improved Abs Rel, Sq Rel, RMSE, and
RMSE log by 5.6%, 35.8%, 42.3%, and 16.5%, respectively,
compared with the second-best method. For accuracy metrics
11

<!-- page 12 -->
Table 5: Quantitative comparison of various methods on the EndoMapper dataset. Columns highlighted in
indicate that higher values are better, while those
highlighted in
indicate that lower values are better. The best results are shown in bold, and the second-best results are underlined.
Method
PSNR
SSIM
LPIPS
Abs Rel
Sq Rel
RMSE
RMSE log
δ < 1.25
δ < 1.252
EndoMapper-Sequence1
EndoNeRF
11.192
0.626
0.824
0.401
30.472
64.341
0.602
0.301
0.589
EndoSurf
23.767
0.689
0.714
7.182
740.954
62.653
0.962
0.258
0.494
ForPlane
27.026
0.811
0.552
0.825
24.607
49.890
0.622
0.283
0.603
DDS-SLAM
22.315
0.739
0.643
0.443
28.178
40.461
0.594
0.364
0.623
ENeRF-SLAM
19.661
0.757
0.609
0.499
31.652
44.688
0.700
0.298
0.582
EndoGSLAM
17.452
0.675
0.551
0.971
88.893
63.412
0.886
0.197
0.392
Endo-2DTAM
15.509
0.583
0.622
0.988
90.022
62.122
0.869
0.209
0.413
Endo-4DGS
19.453
0.714
0.682
0.615
9.388
28.909
0.630
0.786
0.930
SurgicalGaussian
20.450
0.720
0.572
0.669
10.803
29.715
0.680
0.738
0.872
LGS
23.560
0.772
0.555
0.713
12.597
30.475
0.692
0.655
0.848
EndoGaussian
25.137
0.792
0.583
0.583
8.911
25.999
0.441
0.759
0.913
Deform3DGS
24.767
0.788
0.539
0.615
9.388
28.909
0.630
0.786
0.930
EH-SurGS
25.477
0.803
0.572
0.610
9.373
28.722
0.618
0.794
0.936
Local-EndoGS
33.483
0.944
0.198
0.136
3.265
8.422
0.152
0.903
0.973
EndoMapper-Sequence2
EndoNeRF
11.032
0.625
0.813
0.784
72.588
65.078
0.857
0.195
0.393
EndoSurf
24.445
0.703
0.703
8.066
661.562
59.846
1.013
0.267
0.510
ForPlane
27.075
0.811
0.554
1.096
28.376
47.761
0.626
0.297
0.622
DDS-SLAM
21.916
0.747
0.631
0.444
27.051
39.969
0.626
0.358
0.614
ENeRF-SLAM
19.118
0.755
0.611
0.512
31.669
44.348
0.731
0.284
0.560
EndoGSLAM
17.290
0.677
0.562
0.903
83.272
64.273
0.882
0.203
0.387
Endo-2DTAM
15.686
0.592
0.615
0.962
83.923
62.772
0.868
0.209
0.399
Endo-4DGS
17.827
0.673
0.697
0.635
10.086
31.252
0.677
0.742
0.847
SurgicalGaussian
20.033
0.708
0.628
0.696
11.148
33.063
0.684
0.725
0.853
LGS
23.053
0.752
0.579
0.728
12.809
35.726
0.708
0.696
0.833
EndoGaussian
24.870
0.790
0.586
0.595
9.317
29.562
0.459
0.767
0.892
Deform3DGS
24.250
0.773
0.511
0.635
10.086
31.252
0.677
0.742
0.847
EH-SurGS
24.847
0.795
0.566
0.628
10.055
31.109
0.661
0.747
0.864
Local-EndoGS
32.993
0.940
0.217
0.145
3.640
9.542
0.163
0.885
0.961
EndoMapper-Sequence3
EndoNeRF
9.703
0.572
0.825
0.898
88.927
67.545
0.912
0.186
0.365
EndoSurf
25.618
0.748
0.663
7.947
1013.194
56.154
0.906
0.262
0.499
ForPlane
26.979
0.814
0.569
1.077
24.212
44.934
0.605
0.335
0.646
DDS-SLAM
22.315
0.739
0.643
0.469
31.711
41.428
0.624
0.347
0.603
ENeRF-SLAM
18.869
0.759
0.619
0.541
32.177
44.780
0.751
0.249
0.519
EndoGSLAM
11.755
0.665
0.566
1.000
93.154
63.095
0.901
0.204
0.392
Endo-2DTAM
15.120
0.579
0.621
0.998
87.916
61.317
0.876
0.214
0.411
Endo-4DGS
19.453
0.714
0.682
0.619
9.671
29.755
0.658
0.764
0.918
SurgicalGaussian
20.450
0.720
0.572
0.652
9.972
30.933
0.675
0.761
0.896
LGS
23.560
0.772
0.555
0.659
11.288
33.825
0.687
0.679
0.842
EndoGaussian
25.137
0.792
0.583
0.601
9.344
27.165
0.431
0.783
0.961
Deform3DGS
24.767
0.788
0.539
0.619
9.671
29.755
0.658
0.764
0.918
EH-SurGS
25.477
0.803
0.572
0.612
9.652
29.602
0.639
0.774
0.932
Local-EndoGS
33.483
0.944
0.198
0.141
3.360
8.969
0.159
0.898
0.970
(δ < 1.25 and δ < 1.252), the improvements were 5.0% and
5.6%. The improvements on the Cutting sequence were even
more significant, with error metric improvements ranging from
17.2% to 51.6%, and accuracy metrics improved by 10.0% and
3.3%. These results demonstrate that our approach effectively
preserves high-fidelity appearance while significantly improv-
ing geometric accuracy compared to existing methods.
5.2. Results on the StereoMIS Dataset
Unlike the EndoNeRF dataset, where the camera view is
fixed, the StereoMIS dataset involves a moving camera around
the tissue. This movement introduces additional challenges for
12

<!-- page 13 -->
Figure 6: Qualitative comparison of appearance (RGB) and geometric (depth) reconstruction results produced by Local-EndoGS and existing methods on deformable
scenes from the EndoMapper dataset. Our method produces more accurate and visually consistent results for both appearance and depth, with fewer artifacts and
better delineation of anatomical structures compared to existing approaches.
Table 6: Training Time and Rendering Speed on Different Datasets
Method
EndoNeRF dataset
StereoMIS dataset
Train Time
Rendering Speed
Train Time
Rendering Speed
EndoNeRF
∼8 h
0.18 fps
∼18 h
0.13 fps
EndoSurf
∼8 h
0.18 fps
∼10 h
0.26 fps
Forplane
3.00 min
1.40 fps
2.80 min
1.30 fps
Endo-GS
4.43 min
118.50 fps
6.55 min
85.33 fps
Deform3DGS
1.39 min
354.80 fps
3.28 min
202.17 fps
SurgicalGaussian
2.74 min
159.33 fps
4.49 min
114.00 fps
LGS
1.97 min
135.50 fps
3.70 min
86.83 fps
EndoGaussian
2.89 min
203.67 fps
4.14 min
221.50 fps
EH-SurGS
1.66 min
371.33 fps
3.67 min
215.33 fps
Local-EndoGS
2.38 min
371.00 fps
8.36 min
329.83 fps
deformable reconstruction.
As shown in Table 4, all base-
line methods show a clear decrease in appearance rendering
quality. For example, the PSNR of EndoGaussian Liu et al.
(2025) drops from 37.190 dB on the EndoNeRF-Pulling se-
quence (Table 3) to 20.440 dB on StereoMIS-Sequence1. This
decrease is mainly due to camera movement, which breaks
the static camera assumption and causes inconsistencies be-
tween the observed and canonical spaces. Local-EndoGS con-
sistently achieves the highest PSNR and SSIM, with improve-
ments over the second-best method of 24.1% and 17.1% in
Sequence1, and 31.6% and 22.4% in Sequence2.
For per-
ceptual similarity (LPIPS), Local-EndoGS reduces the error
by 36.8% in Sequence1 and 48.3% in Sequence2 compared
to the next best method, indicating an advantage in both im-
age synthesis fidelity and perceptual quality.
For geometric
reconstruction, Local-EndoGS achieves substantial improve-
ments across all error metrics (Abs Rel, Sq Rel, RMSE, and
RMSE log). In Sequence1, improvements are 68.5%, 72.4%,
72.7%, and 62.9%, respectively. In Sequence2, the improve-
ments are 53.9%, 65.8%, 68.2%, and 53.8%. For accuracy met-
rics (δ < 1.25 and δ < 1.252), Sequence1 shows improvements
of 73.0% and 31.9%, and Sequence2 shows improvements of
63.8% and 16.8%.
We also provide a qualitative comparison of different meth-
ods on the StereoMIS dataset at three time points, illustrat-
ing how the observed deformable scene changes as the cam-
era moves, as shown in Fig. 5. For appearance rendering, most
baseline methods produce visible artifacts or blurring and fail to
capture fine tissue textures. In terms of geometric reconstruc-
tion, as seen in the depth maps, many baseline methods do not
produce reliable results when the camera moves, often result-
ing in noisy or distorted depth estimates. While implicit neural
representation methods (such as EndoSurf, Forplane, and DDS-
SLAM) offer some improvements in fitting, they still lack geo-
metric detail. Moreover, these methods require longer training
times and slower inference speeds (see Sec. 5.4). In contrast,
our Local-EndoGS method preserves finer texture details and
generates more accurate geometric structures, enabling faithful
reconstruction of both appearance and geometry in deformable
surgical environments.
5.3. Results on EndoMapper Dataset
Table 5 presents the quantitative results for all methods on the
three sequences of the EndoMapper dataset. Consistent with the
findings on the StereoMIS dataset, our proposed Local-EndoGS
achieves the best performance across all evaluation metrics. For
image quality metrics such as PSNR, SSIM, and LPIPS, Local-
EndoGS consistently outperforms the second-best method, pro-
13

<!-- page 14 -->
Table 7: Ablation Studies of Different Model Components on the StereoMIS dataset.
Method
PSNR
SSIM
Abs Rel
Sq Rel
RMSE
RMSE log
δ < 1.25
δ < 1.252
FPS
Progressive Window-based Global Scene Representation (PWGSR)
w/o windows
21.76
0.713
1.105
38.888
32.925
0.692
0.354
0.617
306
w/o AWP
31.09
0.910
0.137
4.621
9.674
0.170
0.891
0.949
325
Local Canonical Space Initialization (LCSI)
w SD
31.77
0.923
0.124
3.765
9.249
0.153
0.917
0.975
325
w/o LCSI
28.38
0.885
0.296
9.858
18.794
0.426
0.657
0.814
325
w/o TAP
27.47
0.855
0.281
9.384
14.712
0.384
0.696
0.869
330
w/o CWIP
31.22
0.919
0.133
3.943
9.305
0.157
0.907
0.962
325
w/o EGRR
30.90
0.901
0.139
4.787
9.794
0.170
0.883
0.948
329
Loss Functions (LF)
w/o TL
31.16
0.918
0.129
3.921
9.285
0.156
0.908
0.964
326
w/o PBR
30.87
0.897
0.138
4.764
9.983
0.174
0.879
0.942
322
Full model
31.49
0.922
0.129
3.852
9.271
0.154
0.912
0.967
330
ducing images with fewer artifacts and better preservation of
fine details. In terms of depth estimation accuracy—including
Abs Rel, RMSE, and RMSE log—Local-EndoGS demonstrates
clear improvements, delivering more reliable and accurate ge-
ometric reconstructions compared to other approaches.
For
methods specifically designed for colonoscopy videos, such as
EndoGSLAM Wang et al. (2024a) and Endo-2DTAM Huang
et al. (2025), their performance is noticeably inferior since they
ignore the deformable nature of the scene. Qualitative compar-
isons are shown in Fig. 6.
5.4. Analysis of Model Efficiency
To evaluate model efficiency, we report the training time and
inference speed (frames per second, FPS) for each method on
the EndoNeRF and StereoMIS datasets in Table 6. On the En-
doNeRF dataset, our method completes training in about two
minutes, which is comparable to most 3DGS-based methods,
and is suitable for clinical analysis and offline processing. It
achieves a real-time rendering speed of 371 FPS, supporting
efficient visualization and post-processing. On the StereoMIS
dataset, the training time increases due to larger scene sizes
(over 2000 frames), wider camera movement, and more com-
plex tissue deformation. These factors require more local win-
dows (as shown in Table 1) and additional optimization steps.
This increase is expected, as more extensive and dynamic sur-
gical scenes demand extra computation to capture local varia-
tions and ensure reliable reconstruction. Notably, our method
achieves the fastest inference speed, demonstrating high effi-
ciency and scalability.
5.5. Ablation Study
In this subsection, we evaluate the effectiveness of each mod-
ule in our proposed method through comprehensive ablation ex-
periments on StereoMIS-Sequence2. The quantitative results
are summarized in Table 7.
Progressive Window-based Global Scene Representation
(PWGSR). To model long endoscopic sequences with arbitrary
camera motion, we use adaptive window partitioning (AWP)
Figure 7: Qualitative comparison of ablation results for RGB reconstruction
(top row) and depth estimation (bottom row). (a) Without local windows; (b)
Without local canonical space initialization (LCSI); (c) Full model; (d) Refer-
ence.
to divide the input sequence into multiple local windows. We
then progressively optimize the parameters of each local model
to represent the scene within each window. To assess the ef-
fectiveness of this approach, we conduct three experiments:
(1) the full model; (2) a variant without window partition-
ing (w/o windows), which uses a single window for the entire
scene, similar to previous methods; and (3) a variant that uses
uniform window partitioning instead of AWP (w/o AWP). The
number of windows is kept the same as in the full model to
ensure a fair comparison. As shown in Table 7, removing win-
dow partitioning leads to a clear drop in performance because a
single canonical space and deformation network cannot capture
the full range of scene deformations. Fig. 7 further demon-
strates this effect. The results from the w/o windows variant
(Fig. 7(a)) show pronounced artifacts and loss of structural de-
tail, while the full model (Fig. 7(c)) produces reconstructions
that closely match the reference. Similarly, removing the AWP
component (w/o AWP) also reduces performance, as uniform
partitioning uses a fixed window size and cannot adapt to re-
gions with varying dynamics.
Local Canonical Space Initialization (LCSI). We conduct
six ablation experiments to evaluate the effectiveness of LCSI:
(1) the full model; (2) initialization using a stereo depth prior,
as in existing methods (w SD); (3) removing local canonical
14

<!-- page 15 -->
Figure 8: Visual ablation study on the impact of Physics-Based Regulariza-
tion (PBR). The top row shows the rendering results of the model without PBR
(w/o PBR), while the bottom row presents the outputs of the full model. Re-
gions highlighted in yellow indicate areas where the absence of PBR leads to
visual artifacts, demonstrating that PBR helps preserve structural consistency
and anatomical fidelity.
space initialization and using only a monocular depth prior
(w/o LCSI); (4) replacing the TAP model with SIFT keypoints
Lowe (2004) and brute-force matching from OpenCV to es-
tablish inter-frame correspondences (w/o TAP); (5) removing
cross-window information propagation (w/o CWIP); and (6)
removing Error-Guided Region Refinement (w/o EGRR). As
shown in Table 7, although our full model uses monocular in-
put, it achieves performance comparable to the model using
stereo depth priors (w SD). This demonstrates the effectiveness
of our initialization scheme for monocular sequences. Further-
more, initializing with only a monocular depth prior (w/o LCSI)
leads to a significant drop in performance, mainly due to scale
ambiguity. As shown in Fig. 7(b), the reconstructed RGB image
has noticeable artifacts, with many anatomical details blurred.
In addition, the estimated depth map is less accurate compared
to the full model (Fig. 7(c)) and the reference (Fig. 7(d)). In
contrast, our full model produces both RGB and depth results
that closely resemble the reference, highlighting the effective-
ness of the proposed local canonical space initialization. The
results of removing the TAP model show that traditional cor-
respondence methods cannot effectively address the challenges
in endoscopic images, underscoring the necessity of the TAP
module. Removing CWIP also reduces performance, as it helps
maintain spatial consistency between different local windows.
Finally, removing EGRR lowers performance, confirming that
error-guided region refinement corrects regional inaccuracies
and improves overall reconstruction quality.
Loss Functions (LF). We introduce 2D Tracking Loss and
Physics-Based Regularization to enhance our algorithm.
To
assess their effectiveness, we conduct three experiments: the
full model, the model without 2D Tracking Loss (w/o TL), and
the model without Physics-Based Regularization (w/o PBR).
The results in Table 7 show that removing either TL or PBR
reduces performance. This indicates that both losses provide
useful constraints that improve reconstruction quality. In addi-
tion, the visual comparisons presented in Fig. 8 further demon-
strate the effectiveness of the Physics-Based Regularization. As
shown in the figure, the model trained without PBR (w/o PBR)
exhibits noticeable artifacts such as floating noise (highlighted
Table 8: Effect of Monocular Depth Supervision
Method
PSNR
Abs Rel
RMSE
Train Time
w/o Depth Sup.
31.49
0.129
9.271
8.46 min
w/ Depth Sup.
31.41
0.128
9.293
9.92 min
in yellow). In contrast, the full model, guided by physical pri-
ors, produces sharper details and more anatomically consistent
reconstructions.
5.6. Effect of Monocular Depth Supervision
We investigate how using monocular depth maps as a super-
vision signal affects the performance of our method. As de-
scribed in (8) and (9), we obtain the aligned depth map Dfine.
Following previous work, we add an L1 depth loss during train-
ing. We perform experiments on Sequence2 of the StereoMIS
dataset and report quantitative results in Table 8. Our results
show that monocular depth supervision achieves nearly the
same performance as training without it, but slightly increases
the training time (from 8.46 to 9.92 minutes). This suggests
that, at present, monocular depth supervision does not clearly
improve performance. We believe this is because the Depth
Anything model Yang et al. (2024c) is trained on natural scenes,
which limits its ability to generalize to surgical environments.
5.7. Hyperparameters
To assess the impact of varying loss component weights on
the performance of Local-EndoGS, we trained the proposed
framework using multiple weight configurations on the Stere-
oMIS dataset. The quantitative results are illustrated in Fig. 9.
Photometric Loss Weight (λrgb): As evidenced by the re-
sults, increasing λrgb from 0.01 to 1.00 yields a continuous im-
provement in rendering quality, characterized by higher PSNR
and lower RMSE values. This indicates that photometric con-
sistency is fundamental to reconstruction quality, with higher
weights encouraging the model to better capture fine image de-
tails. Consequently, we set λrgb = 1.0 to ensure optimal render-
ing performance.
Tracking Loss Weight (λtrack): The experiments demon-
strate that the model achieves the best trade-off between PSNR
and RMSE at λtrack = 0.010. A weight that is too small (0.001)
fails to provide sufficient geometric guidance, whereas an ex-
cessively large weight (0.100) makes the optimization overly
sensitive to noise in the 2D trajectory prior. This leads to geo-
metric distortions and, consequently, degrades rendering qual-
ity.
Physical Regularization Weights (λrigid, λrot, λiso):
The
rigidity, rotation, and isometry regularization terms exhibit
highly consistent trends. The performance curves display dis-
tinct peaks (for PSNR) or troughs (for RMSE) in the central
region of the parameter space. At lower weights (0.005), the
physical constraints are insufficient, resulting in non-physical
artifacts within the deformation field.
At higher weights
(0.500), excessive smoothing constraints restrict the flexibility
of the Gaussian primitives, leading to a loss of high-frequency
details. The experiments identify 0.050 as the optimal weight
15

<!-- page 16 -->
Figure 9: Illustrations of the performance comparison with different hyperparameter configurations on the StereoMIS dataset.
value. At this setting, the model maintains physical plausibility
without compromising image reconstruction quality.
6. Conclusion
In this work, we present Local-EndoGS, a high-quality 4D
reconstruction framework for deformable surgical scenes from
monocular endoscopic sequences with arbitrary camera move-
ments. Our approach combines a progressive window-based
global representation, a local deformable scene representation,
and a robust coarse-to-fine initialization strategy to effectively
model complex tissue deformations and large camera motions.
We also integrate long-range 2D pixel trajectory constraints and
physical motion priors to improve the accuracy and physical va-
lidity of the reconstructed scenes. Extensive experiments on
multiple datasets, including EndoNeRF, StereoMIS, and En-
doMapper, show that Local-EndoGS achieves better perfor-
mance than existing methods. Our framework has the poten-
tial to support various medical applications, including surgical
planning and clinical training.
Limitations and future work.
Our method has several
limitations that should be addressed in future research. First,
because our framework is based on 3D Gaussian Splatting
(3DGS), it inherits the limitation that 3D Gaussians can-
not accurately represent surfaces due to multi-view inconsis-
tency Huang et al. (2024a). This restricts the accuracy of ge-
ometric reconstruction and the recovery of fine details. In the
future, we will explore ways to improve the multi-view consis-
tency of 3D representations, such as integrating advanced sur-
face regularization techniques Gu´edon and Lepetit (2024) or
developing hybrid representations that combine the strengths
of Gaussians and implicit neural fields Yu et al. (2024). Sec-
ond, like most existing methods for reconstructing deformable
surgical scenes Yang et al. (2024d); Shan et al. (2025a); Li
et al. (2024a); Xie et al. (2024); Liu et al. (2024); Huang et al.
(2024b); Liu et al. (2025), our approach is designed for offline
reconstruction. It produces high-quality deformable reconstruc-
tions that are suitable for treatment planning, surgical educa-
tion, and dataset creation. However, it is not suitable for real-
time applications. In future work, we will focus on improving
computational efficiency and optimizing our algorithms to en-
able real-time deformable reconstruction for intraoperative sur-
gical use. Moreover, our current framework incorporates physi-
cal priors, including a local isometry prior that assumes contin-
uous tissue deformation without topological change and con-
strains surface patches to preserve local geometry, which may
not fully capture events such as cutting or tearing. Extending
the model to handle topological changes (e.g., through adaptive
surface representations or an event detection module) will be
an important direction for future research. In addition, another
limitation concerns our training strategy. Specifically, we first
divide the scene into local windows and then train each window
sequentially. Although this approach allows us to handle long
sequences with arbitrary camera motion and improves recon-
struction performance (as shown in Sec. 5.5), the total training
time increases linearly with the number of windows. More-
over, this sequential process does not fully utilize the parallel
processing capabilities of modern GPUs, leading to subopti-
mal computational efficiency. In the future, we will develop
parallel training strategies to process multiple local windows
simultaneously and design more refined inter-window consis-
tency mechanisms. This will reduce overall training time and
make better use of GPU resources. Finally, incorporating spe-
cialized optical-flow or multi-view 3D matching models (e.g.,
MFT Neoral et al. (2024), MASt3R Leroy et al. (2024)) into
our initialization framework represents another promising di-
rection for future work. Exploring the establishment of corre-
spondences through these methods and leveraging their respec-
tive strengths could further improve the quality and robustness
of our reconstruction framework.
7. Declaration of Competing Interest
The authors declare that they have no known competing fi-
nancial interests or personal relationships that could have ap-
peared to influence the work reported in this paper.
8. Acknowledgement
This work was supported in part by the National Key
R&D Program of China (Grant No.2023YFB4705700), in
part by the Natural Science Foundation of China under Grant
62225309, U24A20278, 62361166632, U21A20480, 6240331
and 62203298, in part by State Key Laboratory of Robotics
and Intelligent Systems (No: 2024-O26), in part by Innova-
tion and Technology Commission of Hong Kong (ITS/235/22,
ITS/225/23, ITS/224/23, MHP/096/22 and Multi-scale Medi-
cal Robotics Center, InnoHK), and in part by Research Grants
Council of Hong Kong (CUHK 14217822, CUHK 14207823,
CUHK 14211425, T45-401/22-N and AoE/E-407/24-N).
References
Azagra, P., Sostres, C., Ferr´andez, ´A., Riazuelo, L., Tomasini, C., Barbed, O.L.,
Morlana, J., Recasens, D., Batlle, V.M., G´omez-Rodr´ıguez, J.J., et al., 2023.
Endomapper dataset of complete calibrated endoscopy procedures. Scien-
tific Data 10, 671.
16

<!-- page 17 -->
Batlle, V.M., Montiel, J.M., Fua, P., Tard´os, J.D., 2023. Lightneus: Neural
surface reconstruction in endoscopy using illumination decline, in: Inter-
national Conference on Medical Image Computing and Computer-Assisted
Intervention, Springer. pp. 502–512.
Bonilla, S., Zhang, S., Psychogyios, D., Stoyanov, D., Vasconcelos, F., Bano,
S., 2024. Gaussian pancakes: geometrically-regularized 3d gaussian splat-
ting for realistic endoscopic reconstruction, in: International Conference on
Medical Image Computing and Computer-Assisted Intervention, Springer.
pp. 274–283.
Cheema, M.N., Nazir, A., Sheng, B., Li, P., Qin, J., Kim, J., Feng, D.D., 2018.
Image-aligned dynamic liver reconstruction using intra-operative field of
views for minimal invasive surgery. IEEE Transactions on Biomedical En-
gineering 66, 2163–2173.
Chen, L., Tang, W., John, N.W., Wan, T.R., Zhang, J.J., 2018. Slam-based
dense surface reconstruction in monocular minimally invasive surgery and
its application to augmented reality. Computer methods and programs in
biomedicine 158, 135–146.
Chen, W., Chen, L., Wang, R., Pollefeys, M., 2024.
Leap-vo: Long-term
effective any point tracking for visual odometry, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
19844–19853.
Cui, Z., He, Y., Zhang, P., Hu, Y., Jin, H., Liu, S., 2021. Virtual reality navi-
gation system of nasal endoscopy with real surface texture information, in:
2021 IEEE International Conference on Real-time Computing and Robotics
(RCAR), IEEE. pp. 135–140.
Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A., 2023.
K-planes: Explicit radiance fields in space, time, and appearance, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 12479–12488.
Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.,
2022. Plenoxels: Radiance fields without neural networks, in: Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 5501–5510.
Gu´edon, A., Lepetit, V., 2024. Sugar: Surface-aligned gaussian splatting for
efficient 3d mesh reconstruction and high-quality mesh rendering, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 5354–5363.
Guo, J., Wang, J., Kang, D., Dong, W., Wang, W., Liu, Y.h., 2024a. Free-surgs:
Sfm-free 3d gaussian splatting for surgical scene reconstruction, in: Inter-
national Conference on Medical Image Computing and Computer-Assisted
Intervention, Springer. pp. 350–360.
Guo, J., Wang, J., Wei, R., Kang, D., Dou, Q., Liu, Y.h., 2024b. Uc-nerf:
Uncertainty-aware conditional neural radiance fields from endoscopic sparse
views. IEEE Transactions on Medical Imaging .
Hayoz, M., Hahne, C., Gallardo, M., Candinas, D., Kurmann, T., Allan, M.,
Sznitman, R., 2023. Learning how to robustly estimate camera pose in en-
doscopic videos. International journal of computer assisted radiology and
surgery 18, 1185–1192.
Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S., 2024a. 2d gaussian splat-
ting for geometrically accurate radiance fields, in: ACM SIGGRAPH 2024
conference papers, pp. 1–11.
Huang, Y., Cui, B., Bai, L., Chen, Z., Wu, J., Li, Z., Liu, H., Ren, H., 2025. Ad-
vancing dense endoscopic reconstruction with gaussian splatting-driven sur-
face normal-aware tracking and mapping. arXiv preprint arXiv:2501.19319
.
Huang, Y., Cui, B., Bai, L., Guo, Z., Xu, M., Islam, M., Ren, H., 2024b.
Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian
splatting, in: International Conference on Medical Image Computing and
Computer-Assisted Intervention, Springer. pp. 197–207.
˙Incetan, K., Celik, I.O., Obeid, A., Gokceler, G.I., Ozyoruk, K.B., Almalioglu,
Y., Chen, R.J., Mahmood, F., Gilbert, H., Durr, N.J., et al., 2021. Vr-caps:
a virtual environment for capsule endoscopy. Medical image analysis 70,
101990.
Karaev, N., Rocco, I., Graham, B., Neverova, N., Vedaldi, A., Rupprecht, C.,
2024. Cotracker: It is better to track together, in: European conference on
computer vision, Springer. pp. 18–35.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., Drettakis, G., 2023. 3d gaussian splat-
ting for real-time radiance field rendering. ACM Trans. Graph. 42, 139–1.
Lamarca, J., Parashar, S., Bartoli, A., Montiel, J., 2020. Defslam: Tracking and
mapping of deforming scenes from monocular sequences. IEEE Transac-
tions on robotics 37, 291–303.
Leroy, V., Cabon, Y., Revaud, J., 2024. Grounding image matching in 3d with
mast3r, in: European Conference on Computer Vision, Springer. pp. 71–91.
Li, C., Feng, B.Y., Liu, Y., Liu, H., Wang, C., Yu, W., Yuan, Y., 2024a. En-
dosparse: Real-time sparse view synthesis of endoscopic scenes using gaus-
sian splatting, in: International Conference on Medical Image Computing
and Computer-Assisted Intervention, Springer. pp. 252–262.
Li, H., Shan, J., Wang, H., 2024b. Sdfplane: Explicit neural surface reconstruc-
tion of deformable tissues, in: International Conference on Medical Image
Computing and Computer-Assisted Intervention, Springer. pp. 542–552.
Li, Y., Richter, F., Lu, J., Funk, E.K., Orosco, R.K., Zhu, J., Yip, M.C., 2020.
Super: A surgical perception framework for endoscopic tissue manipulation
with surgical robotics. IEEE Robotics and Automation Letters 5, 2294–
2301.
Li, Z., Liu, X., Drenkow, N., Ding, A., Creighton, F.X., Taylor, R.H., Unberath,
M., 2021. Revisiting stereo depth estimation from a sequence-to-sequence
perspective with transformers, in: Proceedings of the IEEE/CVF interna-
tional conference on computer vision, pp. 6197–6206.
Liu, H., Liu, Y., Li, C., Li, W., Yuan, Y., 2024. Lgs: A light-weight 4d gaussian
splatting for efficient surgical scene reconstruction, in: International Con-
ference on Medical Image Computing and Computer-Assisted Intervention,
Springer. pp. 660–670.
Liu, Y., Li, C., Liu, H., Yang, C., Yuan, Y., 2025. Foundation model-guided
gaussian splatting for 4d reconstruction of deformable tissues. IEEE Trans-
actions on Medical Imaging .
Long, Y., Li, Z., Yee, C.H., Ng, C.F., Taylor, R.H., Unberath, M., Dou,
Q., 2021.
E-dssr: efficient dynamic surgical scene reconstruction with
transformer-based stereoscopic depth perception, in: Medical Image Com-
puting and Computer Assisted Intervention–MICCAI 2021: 24th Interna-
tional Conference, Strasbourg, France, September 27–October 1, 2021, Pro-
ceedings, Part IV 24, Springer. pp. 415–425.
Lowe, D.G., 2004. Distinctive image features from scale-invariant keypoints.
International journal of computer vision 60, 91–110.
Luiten, J., Kopanas, G., Leibe, B., Ramanan, D., 2024. Dynamic 3d gaussians:
Tracking by persistent dynamic view synthesis, in: 2024 International Con-
ference on 3D Vision (3DV), IEEE. pp. 800–809.
Ma, R., Wang, R., Zhang, Y., Pizer, S., McGill, S.K., Rosenman, J., Frahm,
J.M., 2021.
Rnnslam: Reconstructing the 3d colon to visualize missing
regions during a colonoscopy. Medical image analysis 72, 102100.
Mahmoud, N., Cirauqui, I., Hostettler, A., Doignon, C., Soler, L., Marescaux,
J., Montiel, J.M.M., 2017. Orbslam-based endoscope tracking and 3d re-
construction, in: Computer-Assisted and Robotic Endoscopy: Third Inter-
national Workshop, CARE 2016, Held in Conjunction with MICCAI 2016,
Athens, Greece, October 17, 2016, Revised Selected Papers 3, Springer. pp.
72–83.
Mahmoud, N., Collins, T., Hostettler, A., Soler, L., Doignon, C., Montiel,
J.M.M., 2018. Live tracking and dense reconstruction for handheld monoc-
ular endoscopy. IEEE transactions on medical imaging 38, 79–89.
Maier-Hein, L., et al., 2017. Surgical data science for next-generation interven-
tional medicine. Nature Biomedical Engineering 1, 691–696.
Marmol, A., Banach, A., Peynot, T., 2019. Dense-arthroslam: Dense intra-
articular 3-d reconstruction with robust localization prior for arthroscopy.
IEEE Robotics and Automation Letters 4, 918–925.
M¨uller, T., Evans, A., Schied, C., Keller, A., 2022. Instant neural graphics prim-
itives with a multiresolution hash encoding. ACM transactions on graphics
(TOG) 41, 1–15.
Neoral, M., ˇSer`ych, J., Matas, J., 2024. Mft: Long-term tracking of every
pixel, in: Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision, pp. 6837–6847.
Ota, D., Loftin, B., Saito, T., Lea, R., Keller, J., 1995. Virtual reality in surgical
education. Computers in biology and medicine 25, 127–137.
Schmidt, A., Mohareri, O., DiMaio, S., Yip, M.C., Salcudean, S.E., 2024.
Tracking and mapping in medical computer vision: A review. Medical Im-
age Analysis , 103131.
Schonberger, J.L., Frahm, J.M., 2016. Structure-from-motion revisited, in: Pro-
ceedings of the IEEE conference on computer vision and pattern recogni-
tion, pp. 4104–4113.
Sengupta, A., Bartoli, A., 2021. Colonoscopic 3d reconstruction by tubular
non-rigid structure-from-motion.
International Journal of Computer As-
sisted Radiology and Surgery 16, 1237–1241.
Shan, J., Cai, Z., Hsieh, C.T., Cheng, S.S., Wang, H., 2025a.
Deformable
gaussian splatting for efficient and high-fidelity reconstruction of surgical
17

<!-- page 18 -->
scenes. arXiv preprint arXiv:2501.01101 .
Shan, J., Li, Y., Xie, T., Wang, H., 2024a. Enerf-slam: A dense endoscopic slam
with neural implicit representation. IEEE Transactions on Medical Robotics
and Bionics .
Shan, J., Li, Y., Yang, L., Feng, Q., Han, L., Wang, H., 2024b. Dds-slam: Dense
semantic neural slam for deformable endoscopic scenes, in: 2024 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS), IEEE.
pp. 10837–10842.
Shan, J., Zhang, Z., Li, H., Hsieh, C.T., Li, Y., Wu, W., Wang, H., 2025b.
Uw-dnerf: Deformable soft tissue reconstruction with uncertainty-guided
depth supervision and local information integration. IEEE Transactions on
Medical Imaging .
Shao, S., Pei, Z., Chen, W., Zhu, W., Wu, X., Sun, D., Zhang, B., 2022. Self-
supervised monocular depth and ego-motion estimation in endoscopy: Ap-
pearance flow to the rescue. Medical image analysis 77, 102338.
Shaw, R., Nazarczuk, M., Song, J., Moreau, A., Catley-Chandar, S., Dhamo,
H., P´erez-Pellitero, E., 2024.
Swings: sliding windows for dynamic 3d
gaussian splatting, in: European Conference on Computer Vision, Springer.
pp. 37–54.
Tewari, A., Fried, O., Thies, J., Sitzmann, V., Lombardi, S., Sunkavalli, K.,
Martin-Brualla, R., Simon, T., Saragih, J., Nießner, M., et al., 2020. State
of the art on neural rendering, in: Computer Graphics Forum, Wiley Online
Library. pp. 701–727.
Tewari, A., Thies, J., Mildenhall, B., Srinivasan, P., Tretschk, E., Yifan, W.,
Lassner, C., Sitzmann, V., Martin-Brualla, R., Lombardi, S., et al., 2022.
Advances in neural rendering, in: Computer Graphics Forum, Wiley Online
Library. pp. 703–735.
Wang, K., Yang, C., Wang, Y., Li, S., Wang, Y., Dou, Q., Yang, X., Shen, W.,
2024a. Endogslam: Real-time dense reconstruction and tracking in endo-
scopic surgeries using gaussian splatting, in: International Conference on
Medical Image Computing and Computer-Assisted Intervention, Springer.
pp. 219–229.
Wang, Q., Ye, V., Gao, H., Austin, J., Li, Z., Kanazawa, A., 2024b.
Shape of motion: 4d reconstruction from a single video. arXiv preprint
arXiv:2407.13764 .
Wang, Y., Long, Y., Fan, S.H., Dou, Q., 2022. Neural rendering for stereo
3d reconstruction of deformable tissues in robotic surgery, in: International
conference on medical image computing and computer-assisted interven-
tion, Springer. pp. 431–441.
Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P., 2004. Image quality
assessment: from error visibility to structural similarity. IEEE transactions
on image processing 13, 600–612.
Wu, T., Miao, Y., Li, Z., Zhao, H., Dang, K., Su, J., Yu, L., Li, H., 2025.
Endoflow-slam: Real-time endoscopic slam with flow-constrained gaussian
splatting, in: International Conference on Medical Image Computing and
Computer-Assisted Intervention, Springer. pp. 202–212.
Xie, D., Duan, X., Ma, L., Zhao, M., Lu, J., Li, C., 2022. Mixed reality as-
sisted orbital reconstruction navigation system for reduction surgery of or-
bital fracture, in: 2022 IEEE International Conference on Real-time Com-
puting and Robotics (RCAR), IEEE. pp. 316–321.
Xie, W., Yao, J., Cao, X., Lin, Q., Tang, Z., Dong, X., Guo, X., 2024. Surgical-
gaussian: Deformable 3d gaussians for high-fidelity surgical scene recon-
struction, in: International Conference on Medical Image Computing and
Computer-Assisted Intervention, Springer. pp. 617–627.
Yang, C., Wang, K., Wang, Y., Dou, Q., Yang, X., Shen, W., 2024a. Efficient
deformable tissue reconstruction via orthogonal neural plane. IEEE Trans-
actions on Medical Imaging .
Yang, C., Wang, K., Wang, Y., Dou, Q., Yang, X., Shen, W., 2024b. Efficient
deformable tissue reconstruction via orthogonal neural plane. IEEE Trans-
actions on Medical Imaging .
Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H., 2024c.
Depth anything v2. Advances in Neural Information Processing Systems
37, 21875–21911.
Yang, S., Li, Q., Shen, D., Gong, B., Dou, Q., Jin, Y., 2024d. Deform3dgs:
Flexible deformation for fast surgical scene reconstruction with gaussian
splatting, in: International Conference on Medical Image Computing and
Computer-Assisted Intervention, Springer. pp. 132–142.
Yu, M., Lu, T., Xu, L., Jiang, L., Xiangli, Y., Dai, B., 2024. Gsdf: 3dgs meets
sdf for improved neural rendering and reconstruction. Advances in Neural
Information Processing Systems 37, 129507–129530.
Zha, R., Cheng, X., Li, H., Harandi, M., Ge, Z., 2023.
Endosurf: Neural
surface reconstruction of deformable tissues with stereo endoscope videos,
in: International Conference on Medical Image Computing and Computer-
Assisted Intervention, Springer. pp. 13–23.
Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O., 2018. The unrea-
sonable effectiveness of deep features as a perceptual metric, in: Proceed-
ings of the IEEE conference on computer vision and pattern recognition, pp.
586–595.
Zhu, L., Wang, Z., Cui, J., Jin, Z., Lin, G., Yu, L., 2024. Endogs: deformable
endoscopic tissues reconstruction with gaussian splatting, in: International
Conference on Medical Image Computing and Computer-Assisted Interven-
tion, Springer. pp. 135–145.
Zwicker, M., Pfister, H., Van Baar, J., Gross, M., 2002. Ewa splatting. IEEE
Transactions on Visualization and Computer Graphics 8, 223–238.
18
