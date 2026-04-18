<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
GaussianSwap: Animatable Video Face Swapping with
3D Gaussian Splatting
Xuan Cheng, Jiahao Rao, Chengyang Li, Wenhao Wang, Weilin Chen, Lvqing Yang
Abstract—We introduce GaussianSwap, a novel video face
swapping framework that constructs a 3D Gaussian Splatting
based face avatar from a target video while transferring identity
from a source image to the avatar. Conventional video swapping
frameworks are limited to generating facial representations in
pixel-based formats. The resulting swapped faces exist merely as
a set of unstructured pixels without any capacity for animation or
interactive manipulation. Our work introduces a paradigm shift
from conventional pixel-based video generation to the creation
of high-fidelity avatar with swapped faces. The framework first
preprocesses target video to extract FLAME parameters, camera
poses and segmentation masks, and then rigs 3D Gaussian splats
to the FLAME model across frames, enabling dynamic facial
control. To ensure identity preserving, we propose an compound
identity embedding constructed from three state-of-the-art face
recognition models for avatar finetuning. Finally, we render the
face-swapped avatar on the background frames to obtain the
face-swapped video. Experimental results demonstrate that Gaus-
sianSwap achieves superior identity preservation, visual clarity
and temporal consistency, while enabling previously unattainable
interactive applications.
Index Terms—video face-swapping, 3DGS, face avatar
I. INTRODUCTION
Face swapping is a technique that transfers identity char-
acteristics from a source image to a subject in a target
image or video while preserving the target’s non-identity
attributes, including pose, facial expressions and background.
This technique holds significant potential for applications in
movie production, digital human and privacy protection. Since
Deepfakes [1] first emerged in 2017 and captured widespread
attention, the field has maintained sustained research interest
and continues to evolve.
Early methods in face swapping primarily concentrated
on still images, such as GAN-based methods [2]–[6] and
Diffusion-based methods [7]–[10]. To process a video, re-
cent methods like DynamicFace [11], VividFace [12] and
HiFiVFS [13] begin to incorporate temporal attention mech-
anismm [14] to maintain temporal continuity across frames.
However, the feature smoothing between frames often com-
promises identity preservation in favor of temporal stability.
Achieving an optimal balance between attribute preservation
and temporal coherence remains a critical issue in this field.
Moreover, both the image and video face swapping methods
are fundamentally limited to generating facial representations
in pixel-based formats, namely images and videos. The result-
ing swapped faces exist merely as a set of unstructured pixels
without any capacity for animation or interactive manipulation.
Xuan Cheng, Jiahao Rao, Chengyang Li, Wenhao Wang, Weilin Chen and
Lvqing Yang are with the School of Informatics, Xiamen University, Xiamen
361005, China. Corresponding author: Lvqing Yang.
Fig. 1.
For video face swapping task, our GaussianSwap can generate not
only face-swapped video (4th row) like conventional methods (2nd row) but
also face-swapped avatar (3rd row), which can facilitate many interactive
applications.
The absence of parametric 3D facial representation in these
pixel-based outputs means they can’t be naturally integrated
into interactive applications that require dynamic facial con-
trol, such as real-time expression editing, novel view rendering
or responsive interaction in virtual environments.
To address the aforementioned challenges in video face
swapping, we propose a paradigm shift from conventional
pixel-based video generation to the creation of high-fidelity
avatars with swapped faces. We show an example in Fig.
1. These head avatars retain all the functionalities of tra-
ditional methods when rendered to image plane, producing
face-swapped videos of competitive temporal stability and
visual quality. Crucially, because the face swapping is per-
formed on the avatars constructed from the target video
frames, temporal coherence is inherently preserved. Moreover,
the inherent animatability of these avatars unlocks a range
of previously unattainable interactive applications, including
video face reenactment, speech/text-driven facial animation
and dynamic background manipulation in video conferenc-
arXiv:2601.05511v1  [cs.CV]  9 Jan 2026

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
ing. These capabilities can’t be achieved by existing video
face swapping methods without extensive, application-specific
post-processing pipelines.
To this end, we propose GaussianSwap, a novel video face
swapping framework that constructs a 3D Gaussian Splatting
(3DGS) [15] based head avatar from a monocular target video
while preserving the identity characteristics extracted from a
source image. Our framework employ 3DGS as the primary
3D head representation due to its unique combination of real-
time rendering and high visual fidelity, which enables seamless
avatar-environment interactions. The pipeline of Gaussian-
Swap begins by preprocessing the target video through 3D face
tracking and head-torso segmentation, extracting per-frame
FLAME [16] parameters, camera poses and segmentation
masks. We then construct an animatable 3DGS-based avatar
by rigging the Gaussian splats to the FLAME parametric face
model across all frames. To ensure identity fidelity between
the avatar and the source image, we further finetune the avatar
through incorporating a compound identity embedding, which
is constructed by three state-of-the-art face recognition models.
The constructed head avatar can be used not only to generate
face-swapped videos, but also to seamlessly integrate into
previously unattainable interactive applications.
The contributions of this paper are as follows:
• We propose a novel video swapping framework that
constructs a high-fidelity 3DGS-based head avatar from
a target video while transferring identity from a source
image to the avatar.
• We propose a novel identity preserving approach in 3DGS
optimization that uses compound identity embedding to
comprehensively capture identity characteristic.
• We show that the face swapping results can be seamlessly
used in several new application scenarios: video face
reenactment, speech-driven facial animation and dynamic
background manipulation.
II. RELATED WORK
A. Image Face Swapping
Image face swapping methods can generally be categorized
into two groups based on their generative models: GAN-based
methods and Diffusion-based methods.
FSGAN [2] pioneered the integration of GANs with face
swapping, introducing a multi-network framework capable of
simultaneous face swapping and reenactment. Subsequently,
SimSwap [3] proposed an identity injection module for
feature-level identity transfer and a weak feature matching
loss to implicitly preserve facial attributes, enabling identity-
agnostic face swapping. In the same year, FaceShifter [4]
introduced a two-stage framework for high-quality face swap-
ping under occlusions. Unlike methods relying solely on face
recognition networks for identity preservation, HifiFace [5]
leveraged 3DMM-based [17] facial geometry to enhance iden-
tity fidelity through shape control. Additionally, advancements
in StyleGAN [18]–[20] significantly propelled face-swapping
techniques. For instance, MegaFS [6] utilized its generator to
achieve megapixel-resolution single-shot face swapping.
Driven by Diffusion model’s generative prowess and train-
ing stability, researchers have recently explored diffusion-
based face-swapping methods. DiffFace [7] first employed
a Diffusion model for face swapping, training an identity-
conditioned DDPM [21] and incorporating a facial expert
during sampling to transfer identities while preserving target
attributes. Building on this, DiffSwap [8] guided diffusion
using identity, landmarks and facial attributes, redefining face
swapping as conditional image inpainting. To jointly handle
face swapping and reenactment, DiffSFSR [9] achieved fine-
grained identity and expression control via diffusion. Most
recently, REFace [10] enhanced identity transfer and attribute
preservation using CLIP [22] features while accelerating high-
quality generation through simplified denoising.
B. Video Face Swapping
While image face swapping methods can process a video
by performing frame-by-frame swapping, they can’t main-
tain temporal consistency. Ghost [23] mitigated the face jit-
tering between adjacent frames through landmark smooth-
ing. Recent advances in video diffusion models [24] have
led to methods like DynamicFace [11], VividFace [12] and
HiFiVFS [13], which employed temporal attention mech-
anism [14] to enhance temporal consistency. Considering
these video Diffusion-based methods often require substantial
computational resources, CanonSwap [25] resolves temporal
instability by decoupling pose variations from identity transfer
in canonical space.
Although these video face swapping methods can pro-
duce stable temporally consistent results, they remain lim-
ited to unstructured pixel representations. In contrast, our
proposed GaussianSwap simultaneously achieves high-quality
video face swapping comparable to state-of-the-art methods,
and generation of dynamically controllable face avatar with
inherent 3D structure.
C. 3D Head Avatar Reconstruction
While 3D face reconstruction [16], [17], [26]–[29] focuses
on the facial region, 3D head avatar reconstruction demands
a comprehensive model of the entire head geometry and its
appearance. Gafni et al. [30] introduced the first 4D head
reconstruction method using Neural Radiance Fields [31], en-
abling expression-driven control through additional expression
conditions in the trained model. The following year, [32] en-
hanced neural representations with an explicit articulated head
model, ensuring high-fidelity reconstruction under significant
viewpoint changes. Concurrently, IMAvatar [33] integrated
blendshapes and skinning into volumetric rendering to manip-
ulate facial expressions and pose deformations. Departing from
earlier methods that relied on 3DMM expression parameters
for driving, LatentAvatar [34] proposed an implicit expression
encoding approach. INSTA [35] leveraged InstantNGP’s [36]
rapid rendering, reducing training time to 10 minutes while
preserving reconstruction quality.
Despite the advantages of implicit modeling, researchers
continue exploring more practical explicit approaches. Begin-
ning with PointAvatar [37] which used point cloud represen-
tations, the emergence of 3DGS [15] has spurred numerous

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
Fig. 2. Overview of the GaussianSwap framework. The framework takes a source image Isrc and a target video Vtgt as input, and generates a high-fidelity
avatar from Vtgt with the face swapped to match Isrc. In the pipeline, FLAME tracking is first performed on the video sequence Vtgt to obtain per-frame
FLAME parameters, camera poses and segmentation/matting masks. A 3DGS-based face avatar is then built using the FLAME tracking data, where the 3D
Gaussians are dynamically bound to the triangular faces of the FLAME mesh models through 3DGS optimization. To enforce the identity similarity between
the avatar and Isrc, the avatar undergoes additional training iterations supervised by three SOTA face recognition models: ArcFace, FaceNet and Dlib. Finally,
the high-fidelity, face-swapped avatar is generated, which can be further rendered into the face-swapped video.
works adopting explicit modeling schemes [38]–[41]. Capi-
talizing on 3DGS’s efficiency, SplattingAvatar [39] achieved
real-time rendering at 300 FPS in GPU and 30 FPS in
mobile device. GaussianAvatars [42] bounds 3D Gaussian
ellipsoids to FLAME [16] meshes during training, enabling
avatar driving through new FLAME inputs. NPGA [38] inte-
grated MonoNPHM’s [43] expression priors for finer-grained
control, while RelightableGaussian [41] employed detailed
appearance modeling (diffuse color, specular highlights and
normals) to achieve relightable rendering. Learn2Talk [44]
and TalkingEyes [45] used speech signals to drive a pre-bulit
3DGS-based head avatar.
III. FRAMEWORK
To reconstruct a high-quality and controllable swapped
dynamic face avatar, GaussianSwap takes a monocular target
video Vtgt and a source face image Isrc as input. The pipeline
of GaussianSwap is illustrated in Fig. 2, which contains four
main steps: target video preprocessing, face avatar reconstruc-
tion, identity finetuning and face-swapped video generation.
A. Target Video Preprocessing
Given a monocular target video, the video preprocessing
step extracts the high-quality 3D face tracking data from it,
thus enabling the creation of animatable face avatar in the next
step.
The preprocessing step firstly applies Robust Video Mat-
ting [46] on each frame of the video to separate the subject
(primarily the head and torso) from the background. Next, the
FLAME [16] tracking produces optimized FLAME parameters
that accurately represent the subject’s facial geometry and
appearance throughout the video sequence. These optimized
FLAME parameters include shape, expression, pose and skin-
ning weights. Similar to DECA [26], the FLAME optimization
employs a loss function comprising facial landmark loss, im-
age reconstruction loss and FLAME parameter regularization.
DECA performs FLAME optimization on only a single
image. When applied to video FLAME tracking, it fails to
maintain consistency in FLAME parameters across frames. To
address this issue, we first conduct frame-by-frame FLAME
optimization, then randomly sample frames from the sequence
to jointly optimize their FLAME parameters for improved
temporal consistency.
B. Face Avatar Reconstruction
We use a binding method similar to GaussianAvatars [42]
to associate 3D Gaussians with the tracked FLAME mesh
models. The 3D Gaussian is initialized to each triangular face
in the local coordinate system by setting the center µ to the
origin, the rotation r to the identity rotation matrix and the
scale s is to the unit vector. Then, these 3D Gaussians are
moved with their parent faces across time steps. Specifically,
the 3D Gaussian is transformed from the local space to the
global space by:
˜r = Kr
˜µ = lKµ + V
˜s = ls,
(1)
where ˜µ, and ˜s denote the center, rotation and scale in the
global space, K and V denote the rotation and translation
transformation matrices computed from each face, and scalar
l represents the size of triangular face. This adaptive scaling l
ensures proportional Gaussian dimensions: larger faces main-
tain association with larger Gaussians and vice versa.
The loss function for optimizing the 3DGS-based face avatar
against the target video comprises a reconstruction loss and a
regularization loss.

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Reconstruction Loss enforces photometric consistency be-
tween the avatar-rendered images It
render and the correspond-
ing target video frames It
tgt. We adopt the L1 loss and the
SSIM loss, which have been widely used in various 3DGS-
based works:
Lrec = (1 −λssim)∥It
render −It
tgt∥1+
λssimSSIM(It
render, It
tgt).
(2)
The hyperparameter λssim that balances the two losses is set
to 0.2.
Regularization Loss regularizes the center µ and the scale
s of 3D Gaussian. Although the reconstruction loss can
generate high-quality rendered images, the lack of additional
constraints may lead to poor alignment between 3D Gaussians
and triangular faces. Therefore, we apply regularization to the
center µ and the scale s of each 3D Gaussian to improve
alignment quality. The regularization loss Lreg consists of two
components: scale loss Lscale and position loss Lpos, which
is defined as:
Lreg = λscaleLscale + λposLpos.
(3)
The hyperparameters λscale and λpos are set to 1 and 0.01
respectively.
The scale loss Lscale is used to constrain the size of each
3D Gaussian, which is formulated as:
Lscale = ∥max(s, ϕscale)∥2.
(4)
ϕscale denotes the threshold, which is set to 0.6. Lscale
encourages the scale ˜s of a 3D Gaussian in the global space to
be no larger than 0.6 times the size of its associated triangular
face. This scaling limitation prevents small triangular surfaces
from being assigned disproportionately large 3D Gaussians,
which could otherwise introduce significant visual artifacts in
avatar animation.
The position loss Lpos maintains spatial coherence by
constraining each 3D Gaussian to remain proximal to its
associated triangular face. For example, a 3D Gaussian initially
bound to the nose should maintain its spatial localization
throughout optimization, preventing erroneous migration to the
eyes. Lpos is defined as:
Lpos = ∥max(µ, ϕpos)∥2,
(5)
where the threshold ϕpos constrains the permissible deviation
of each 3D Gaussian from its associated triangular face, and
it is usually set to 1.0.
C. Identity Finetuning
After the face avatar reconstruction, we obtain the 3DGS-
based face avatar built from the target video. To transfer the
identity from source image to this avatar, we innovatively
incorporate an identity loss based on the compound identity
embedding. The motivation of the compound identity embed-
ding is that a single identity embedding is usually biased and
thus can’t comprehensively capture the identity characteristic.
Three state-of-the-art face recognition models are selected to
construct the compound identity embedding, which includes
Dlib [47], FaceNet [48] and ArcFace [49]. The identity loss
is defined as:
Lid =
K
X
k=1
λk
 1 −cos
 Ek
id (Isrc) , Ek
id
 It
render

,
(6)
where Ek
id (·) represents one of the three (K = 3) identity
encoder constructed by the pretrained face recognition models,
λk controls the contribution of each identity encoder, and
cos(·, ·) denotes the cosine similarity measuring the similar-
ity of two identity embeddings. In our experiments, we set
λ1 = 0.9, λ2 = 0.001 and λ3 = 0.1.
Note that the ArcFace model requires aligned and cropped
face images as input. To avoid additional computational over-
head during optimization, we precompute the necessary affine
transformation matrices in the video preprocessing stage.
The overall loss function in the avatar finetuning stage is
defined as the combination of all the aforementioned losses:
Ltotal = Lrec + Lreg + λidLid,
(7)
where the hyperparameter λid is set to 0.1.
To obtain finer facial details, GaussianSwap adopts a dif-
ferent adaptive density control mechanism compared to the
original 3DGS. It records the indices of the triangular faces
to their associated 3D Gaussians, thereby ensuring that newly
added Gaussians preserve the same binding relationships as
the originals.
D. Face-swapped Video Rendering
To produce the final face-swapped video, the face-swapped
avatar is frame-by-frame rendered on the background im-
ages in the target video. We develop a robust face video
fusion method to seamlessly blend the avatar-rendered images
Iswapped with the background frames Itgt.
Firstly, we conduct face parsing on Iswapped to obtain the
head mask Mswapped. Then, we conduct video matting [46]
on Itgt to obtain the foreground mask Mtgt. Either Mswapped
or Mtgt alone could be used for blending. However, our
experiments show that the combination of them can yield
better blending results. Hence, we define the fused mask
Mfuse as:
Mfuse = Mswapped · Mtgt.
(8)
Due to the sharp boundaries and potential inaccuracies of
Mfuse, directly using Mfuse will introduce visual artifacts
along the blending boundary between the background and the
head. To address this issue, we also conduct edge erosion and
Gaussian smoothing on Mfuse.
Finally, based on Mfuse, we perform a linear combination
of Iswapped and Itgt to obtain the final face-swapped image
in each frame, which is defined as:
Ifinal = Iswapped · Mfuse + Itgt · (1 −Mfuse).
(9)
E. Implementation Details
Most hyperparameters settings have already introduced
in the above subsections, and remain fixed for all inputs
across different subjects and datasets. We use Adam for

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
the 3DGS parameter optimization with the same learning
rates as 3DGS [15] implementation. We reduce the spherical
harmonics order to one. The 3DGS-based face avatar is trained
on a target video for totally 600,000 iterations, followed
by an additional 120,000 iterations for identity finetuning.
Constructing a face-swapped face avatar on an NVIDIA RTX
4090 GPU requires approximately 6 to 10 hours, with duration
varying according to the number of 3D Gaussians (larger
quantities result in longer training times).
IV. EXPERIMENTS
A. Quantitative Evaluation
1) Datasets: We conduct quantitative evaluation on two
datasets: FaceForensics++ (FF++) [50] and INSTA [35]. FF++
is commonly used to evaluate face swapping performance
in images/videos, while INSTA typically assesses head/face
avatar reconstruction quality. From these datasets, we ran-
domly select five monocular videos from FF++ and five from
INSTA as the target videos, along with four portraits of
globally recognized individuals as the source images. This cre-
ates 40 challenging target-source pairs covering same-gender,
cross-gender and cross-ethnicity scenarios. Our GaussianSwap
produces 20 face-swapped avatars on FF++ and 20 face-
swapped avatars on INSTA. To ensure accurate foreground
supervision during training, we use background-removed im-
ages as target frames.
2) Evaluation Metrics: We evaluate the face swapping
performance using both the image-based face swapping met-
rics and the video consistency metrics. The image-related
metrics include identity similarity (IDs), pose error (Pose),
and expression error (Exp), while the video-related metrics
include video identity distance (VIDD). When computing
IDs, since ArcFace [49], FaceNet [48] and Dlib [47] have
been used in GaussianSwap’s training process, we employ
BlendFace [51] as an independent identity feature extractor to
ensure unbiased evaluation. Higher IDs score indicates better
identity preservation. When computing Exp and Pose, we use
Deep3DFaceRecon [52] and HopeNet [53] to estimate expres-
sion and pose parameters for the target video and the swapped
video respectively. We then calculate the mean Euclidean
distance between corresponding parameters to quantify non-
identity attributes preservation. For VIDD, we follow FOS [54]
to assess the temporal consistency between consecutive video
frames. We average the evaluation metric values across all test
data to obtain the overall performance score in each evaluation
metric.
3) Competitors:
We compare our GaussianSwap with
the
image
face
swapping
methods
like
SimSwap
[3],
FaceDancer [55] and E4S [56], and the video face swap-
ping methods like Ghost [23] and CanonSwap [25]. All the
competing methods are open-source and we directly use their
pre-traiend models for evaluation. Since the recent video
face swapping methods including DynamicFace [11], and
HiFiVFS [13] have not released their code or pre-trained
models, we are unable to include them in the quantitative
comparison.
TABLE I
QUANTITATIVE EVALUATION RESULTS ON FF++ DATASET. THE BEST
SCORE IN EACH METRIC IS MARKED WITH BOLD. RED, ORANGE, AND
YELLOW SHADING INDICATE 1ST , 2ND , 3RD
PLACE, RESPECTIVELY.
Methods
Image Quality
Video Quality
IDs↑Pose↓Exp↓
VIDD↓
SimSwap
55.2
1.69
2.41
0.1477
FaceDancer
45.9
2.42
2.81
0.1525
E4S
45.2
3.03
2.99
0.2035
Ghost
70.7
1.95
2.88
0.1504
CanonSwap
62.1
1.58
2.42
0.1510
GaussianSwap
71.7
1.84
2.51
0.1472
TABLE II
QUANTITATIVE EVALUATION RESULTS ON INSTA DATASET. THE BEST
SCORE IN EACH METRIC IS MARKED WITH BOLD. RED, ORANGE, AND
YELLOW SHADING INDICATE 1ST , 2ND , 3RD
PLACE, RESPECTIVELY.
Methods
Image Quality
Video Quality
IDs↑Pose↓Exp↓
VIDD↓
SimSwap
59.8
2.20
2.69
0.3159
FaceDancer
43.3
2.39
2.81
0.3219
E4S
51.0
3.22
3.21
0.4475
Ghost
78.1
2.79
3.06
0.3172
CanonSwap
64.8
1.81
2.54
0.3049
GaussianSwap
81.5
2.06
2.72
0.2808
4) Results: Tab. I and Tab. II present the quantitative eval-
uation results on the FF++ and INSTA datasets, respectively.
Our GaussianSwap demonstrates superior performance com-
pared to competing methods, achieving significantly higher
identity similarity scores on both datasets. These results con-
firm the effectiveness of our compound identity embedding
for identity preservation. Notably, GaussianSwap exhibits out-
standing temporal consistency, achieving the higher video
identity consistency scores on both FF++ and INSTA. By
leveraging all frames from the target video in 3DGS op-
timization for avatar construction, GaussianSwap maintains
exceptional temporal stability throughout the generated se-
quences. Furthermore, GaussianSwap maintains competitive
performance in preserving non-identity attributes (pose and
expression), achieving results comparable to leading methods
while delivering superior identity preservation and temporal
consistency.
B. Qualitative Evaluation
Fig. 3 and Fig. 4 show the qualitative comparisons on
INSTA and FF++, respectively. The results show that our
GaussianSwap not only achieves accurate identity transfer but
also presents high-quality visual appearance in the swapped
faces (e.g., eyebrows, eyes, lips and teeth). Fig. 5 shows
the qualitative comparisons on side-view faces. GaussianSwap
also performs effectively on these challenging poses. The
visual comparisons on video face swapping are presented in
the supplementary video.
C. User Study
For more comprehensive evaluation, we have conducted a
perceptual user study to evaluate the video face swapping qual-
ity of our GaussianSwap and the five baseline methods, which

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
Fig. 3. Qualitative results on INSTA. GaussianSwap achieves accurate identity transfer and high-quality visual appearance in the swapped faces. More details
are visible in the enlarged view.
include SimSwap [3], FaceDancer [55], E4S [56] Ghost [23],
and CanonSwap [25]. Specifically, we adopt the A/B test
in a random order to compare our GaussianSwap with the
baseline methods. In each comparison, the participants were
asked to answer the three questions with A or B: 1) which
resulting video has more similarity with the source image?
2) which resulting video has better temporal consistency? 3)
which resulting video has better visual image quality? We
randomly selected 18 videos from both INSTA and FF++, and
used them to generate 18 A vs. B pairs. Some videos used
in the user study have been presented in the supplementary
video. 15 participants took part in the study, finally yielding
270 entries. We calculated the ratio of participants who prefer
our method over the baseline in each evaluation metric. The
percentage of A/B testing is tabulated in Tab. III, which shows
that participants consistently preferred our method over all
the five baseline methods in terms of identity preservation,
temporal consistency and visual image quality.
TABLE III
USER STUDY RESULTS. THE PERCENTAGE OF ANSWERS WHERE OUR
METHOD IS PREFERRED OVER THE BASELINE METHOD IS LISTED.
Ours vs. Baseline
Identity
Temporal
Visual
Ours vs. SimSwap
76%
56%
82%
Ours vs. FaceDancer
89%
67%
73%
Ours vs. E4S
100%
84%
93%
Ours vs. Ghost
58%
65%
76%
Ours vs. CanonSwap
78%
69%
91%
D. Ablation Study
To demonstrate the effectiveness of the compound identity
embedding, we sequentially remove each identity embedding
TABLE IV
QUANTITATIVE EVALUATION RESULTS IN ABLATION STUDY. THE BEST
SCORE IN EACH METRIC IS MARKED WITH BOLD. RED, ORANGE, AND
YELLOW SHADING INDICATE 1ST , 2ND , 3RD
PLACE, RESPECTIVELY.
Methods
IDs↑
Exp↓Pose↓VIDD↓
w/o a
56.59
2.60
2.17
0.2692
w/o f
81.98
2.58
1.68
0.2896
w/o d
81.34
2.63
1.90
0.2751
full model
82.13
2.57
1.73
0.2746
when computing the identity loss. The three ablation meth-
ods include: 1) w/o a, removing ArcFace model; 2) w/o f,
removing FaceNet model; 3) w/o d, removing Dlib model.
The ablation study is conducted on INSTA. The quantitative
evaluation results are reported in Tab. IV. Compared to the
full model, the removal of any individual embedding results
in significant degradation of identity similarity performance.
Notably, the absence of the ArcFace embedding causes the
most substantial performance drop, confirming its critical role
in identity preservation. To conclude, this quantitative ablation
study confirms the complementary effectiveness of the three
identity embeddings, and validates the necessity of integrating
all of them to achieve optimal video face swapping results.
Fig. 6 shows visual comparisons. As shown in the 2nd row,
removing ArcFace leads to obvious artifacts, color distortions
and inaccurate identity transfer. The 3rd row demonstrates
that removing FaceNet results in degraded image quality with
excessive smoothing and missing facial details (e.g., wrinkles).
In the 4th row, removing Dlib produces noticeable visual flaws,
such as artifacts in the teeth, color inconsistencies around the
eyes and unnatural textures in wrinkles. In contrast, the full

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
Fig. 4. Qualitative results on FF++. GaussianSwap achieves accurate identity transfer and high-quality visual appearance in the swapped faces. More details
are visible in the enlarged view.
Source
Target
Si mSwap
FaceDancer
E4S
CanonSwap
Ghost
Gaussi anSwap( ours)
Fig. 5. Face swapping results for side-view faces on INSTA. GaussianSwap performs effectively on these challenging poses. More details are visible in the
enlarged view.

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
Fig. 6. Qualitative results in ablation study. Removing any individual identity
embedding results in the degradation of image quality.
model delivers the best visual quality and preserves fine facial
details.
E. Downstream Applications
Based on the face-swapped 3DGS avatar generated by our
GaussianSwap, we implement three downstream applications.
The visual results are presented in the supplementary video.
1) Video Face Reenactment.:
This application can be
viewed as the video-driven 3DGS avatar animation. Given a
driven video, we conduct the FLAME tracking [26] which has
been described in the main body of the manuscript to obtain
the camera poses and FLAME parameters. The camera poses,
expression, head rotation and jaw pose parameters in all frames
are transferred into the 3DGS avatar for animation.
2) Speech-driven 3DGS avatar animation.: In this appli-
cation, we use the speech signals to generate the corre-
sponding 3D face motions, which further drive the 3DGS
avatar. We employ the speech-to-motion model proposed in
Learn2Talk [44]. The sequence of mesh models generated by
Learn2Talk is converted into FLAME parameters through 3D
template fitting. These FLAME parameters are then input into
the 3DGS avatar, producing the final 3DGS facial animation.
3) Dynamic Background Manipulation.: When projecting
the 3DGS avatar into the image plane, we can obtain an alpha
channel map for each frame. We then blend the 3DGS avatar
with a new background using alpha compositing.
V. CONCLUSION AND LIMITATION
In this paper, we introduce GaussianSwap, a novel video
face swapping framework that constructs a 3DGS-based head
avatar from a target video while transferring identity from a
source image to the avatar. We believe the idea of empowering
face manipulation through 3D head avatar reconstruction can
inspire advancements in other face generation tasks.
As shown in the quantitative evaluation, our method doesn’t
show advantage in preserving facial expression and pose
attributes from the target video. This is primarily due to
the inherent drawback of FLAME tracking [26], which can’t
extract highly precise expression and pose parameters from the
target video. The continuous progress in the field of 3D face
tracking can help mitigate this issue, such as the application
of L0 optimization techniques [57]–[60] for improved tracking
accuracy.
Additionally, the constructed 3DGS avatar may exhibit
artifacts in side-view rendering if the target video lacks side-
view perspectives. This limitation persists in both previous
NeRF-based methods and concurrent 3DGS-based methods.
Improving the generalization capability to handle dramatically
novel views remains an open research challenge for 3DGS.
REFERENCES
[1] Y. Mirsky and W. Lee, “The creation and detection of deepfakes: A
survey,” ACM Comput. Surv., vol. 54, no. 1, pp. 7:1–7:41, 2022.
[2] Y. Nirkin, Y. Keller, and T. Hassner, “FSGAN: subject agnostic face
swapping and reenactment,” in Proc. of ICCV, 2019, pp. 7183–7192.
[3] R. Chen, X. Chen, B. Ni, and Y. Ge, “Simswap: An efficient framework
for high fidelity face swapping,” in Proc. of ACM MM, 2020, pp. 2003–
2011.
[4] L. Li, J. Bao, H. Yang, D. Chen, and F. Wen, “Advancing high fidelity
identity swapping for forgery detection,” in Proc. of CVPR, 2020, pp.
5073–5082.
[5] Y. Wang, X. Chen, J. Zhu, W. Chu, Y. Tai, C. Wang, J. Li, Y. Wu,
F. Huang, and R. Ji, “Hififace: 3d shape and semantic prior guided high
fidelity face swapping,” in Proc. of IJCAI, 2021, pp. 1136–1142.
[6] Y. Zhu, Q. Li, J. Wang, C. Xu, and Z. Sun, “One shot face swapping
on megapixels,” in Proc. of CVPR, 2021, pp. 4834–4844.
[7] K. Kim, Y. Kim, S. Cho, J. Seo, J. Nam, K. Lee, S. Kim, and K. Lee,
“Diffface: Diffusion-based face swapping with facial guidance,” Pattern
Recognit., vol. 163, p. 111451, 2025.
[8] W. Zhao, Y. Rao, W. Shi, Z. Liu, J. Zhou, and J. Lu, “Diffswap: High-
fidelity and controllable face swapping via 3d-aware masked diffusion,”
in Proc. of CVPR, 2023, pp. 8568–8577.
[9] R. Liu, B. Ma, W. Zhang, Z. Hu, C. Fan, T. Lv, Y. Ding, and X. Cheng,
“Towards a simultaneous and granular identity-expression control in
personalized face generation,” in Proc. of CVPR, 2024, pp. 2114–2123.
[10] S. Baliah, Q. Lin, S. Liao, X. Liang, and M. H. Khan, “Realistic and
efficient face swapping: A unified approach with diffusion models,” in
Proc. of WACV, 2025, pp. 1062–1071.
[11] R. Wang, Y. Chen, S. Xu, T. He, W. Zhu, D. Song, N. Chen, X. Tang,
and Y. Hu, “Dynamicface: High-quality and consistent face swapping
for image and video using composable 3d facial priors,” arXiv preprint
arXiv:2501.08553, 2025.
[12] H. Shao, S. Wang, Y. Zhou, D. H. Guanglu Song, S. Qin, Z. Zong,
B. Ma, Y. Liu, and H. Li, “Vividface: A diffusion-based hybrid
framework for high-fidelity video face swapping,” arXiv preprint
arXiv:2412.11279, 2024.
[13] X. Chen, K. He, J. Zhu, Y. Ge, W. Li, and C. Wang, “Hifivfs: High
fidelity video face swapping,” arXiv preprint arXiv:2411.18293, 2024.
[14] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. of
NeurIPS, 2017, pp. 5998–6008.
[15] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[16] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero, “Learning a model
of facial shape and expression from 4d scans.” ACM Trans. Graph.,
vol. 36, no. 6, pp. 194–1, 2017.
[17] V. Blanz and T. Vetter, “A morphable model for the synthesis of 3d
faces,” in Proc. of SIGGRAPH, 1999, pp. 187–194.

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
[18] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture
for generative adversarial networks,” in Proc. of CVPR, 2019, pp. 4401–
4410.
[19] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila,
“Analyzing and improving the image quality of stylegan,” in Proc of
CVPR, 2020, pp. 8107–8116.
[20] R. Liu, C. Li, H. Cao, Y. Zheng, M. Zeng, and X. Cheng, “EMEF:
ensemble multi-exposure image fusion,” in Proc. of AAAI, B. Williams,
Y. Chen, and J. Neville, Eds., 2023, pp. 1710–1718.
[21] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
in Proc. of NeurIPS, 2020.
[22] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervi-
sion,” in Proc. of ICML, vol. 139, 2021, pp. 8748–8763.
[23] A. Groshev, A. Maltseva, D. Chesakov, A. Kuznetsov, and D. Dimitrov,
“GHOST - A new face swap approach for image and video domains,”
IEEE Access, vol. 10, pp. 83 452–83 462, 2022.
[24] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian,
D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, V. Jampani, and
R. Rombach, “Stable video diffusion: Scaling latent video diffusion
models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
[25] X. Luo, Y. Zhu, Y. Liu, L. Lin, C. Wan, Z. Cai, S.-L. Huang, and
Y. Li, “Canonswap: High-fidelity and consistent video face swapping via
canonical space modulation,” arXiv preprint arXiv:2507.02691, 2025.
[26] Y. Feng, H. Feng, M. J. Black, and T. Bolkart, “Learning an animatable
detailed 3d face model from in-the-wild images,” ACM Trans. Graph.,
vol. 40, no. 4, pp. 88:1–88:13, 2021.
[27] R. Liu, Y. Cheng, S. Huang, C. Li, and X. Cheng, “Transformer-
based high-fidelity facial displacement completion for detailed 3d face
reconstruction,” IEEE Trans. Multim., vol. 26, pp. 799–810, 2024.
[28] C. Li, B. Cheng, Y. Cheng, H. Zhang, R. Liu, Y. Zheng, J. Liao,
and X. Cheng, “Facerefiner: High-fidelity facial texture refinement
with differentiable rendering-based style transfer,” IEEE Trans. Multim.,
vol. 26, pp. 7225–7236, 2024.
[29] H. Cao, B. Cheng, Q. Pu, H. Zhang, B. Luo, Y. Zhuang, J. Lin, L. Chen,
and X. Cheng, “DNPM: A neural parametric model for the synthesis of
facial geometric details,” in Proc. of ICME, 2024, pp. 1–6.
[30] G. Gafni, J. Thies, M. Zollh¨ofer, and M. Nießner, “Dynamic neural
radiance fields for monocular 4d facial avatar reconstruction,” in Proc.
of CVPR, 2021, pp. 8649–8658.
[31] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” in Proc. of ECCV, vol. 12346, 2020, pp. 405–421.
[32] P. Grassal, M. Prinzler, T. Leistner, C. Rother, M. Nießner, and J. Thies,
“Neural head avatars from monocular RGB videos,” in Proc. of CVPR,
2022, pp. 18 632–18 643.
[33] Y. Zheng, V. F. Abrevaya, M. C. B¨uhler, X. Chen, M. J. Black, and
O. Hilliges, “I M avatar: Implicit morphable head avatars from videos,”
in Proc. of CVPR, 2022, pp. 13 535–13 545.
[34] Y. Xu, H. Zhang, L. Wang, X. Zhao, H. Huang, G. Qi, and Y. Liu,
“Latentavatar: Learning latent expression code for expressive neural
head avatar,” in Proc. of SIGGRAPH, 2023, pp. 86:1–86:10.
[35] W. Zielonka, T. Bolkart, and J. Thies, “Instant volumetric head avatars,”
in Proc. of CVPR, 2023, pp. 4574–4584.
[36] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph.,
vol. 41, no. 4, pp. 102:1–102:15, 2022.
[37] Y. Zheng, W. Yifan, G. Wetzstein, M. J. Black, and O. Hilliges,
“Pointavatar: Deformable point-based head avatars from videos,” in Proc
of CVPR, 2023, pp. 21 057–21 067.
[38] S. Giebenhain, T. Kirschstein, M. R¨unz, L. Agapito, and M. Nießner,
“NPGA: neural parametric gaussian avatars,” in Proc. of SIGGRAPH,
2024, pp. 127:1–127:11.
[39] Z. Shao, Z. Wang, Z. Li, D. Wang, X. Lin, Y. Zhang, M. Fan, and
Z. Wang, “Splattingavatar: Realistic real-time human avatars with mesh-
embedded gaussian splatting,” in Proc. of CVPR, 2024, pp. 1606–1616.
[40] J. Xiang, X. Gao, Y. Guo, and J. Zhang, “Flashavatar: High-fidelity head
avatar with efficient gaussian embedding,” in Proc. of CVPR, 2024, pp.
1802–1812.
[41] S. Saito, G. Schwartz, T. Simon, J. Li, and G. Nam, “Relightable
gaussian codec avatars,” in Proc. of CVPR, 2024, pp. 130–141.
[42] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and
M. Nießner, “Gaussianavatars: Photorealistic head avatars with rigged
3d gaussians,” in Proc. of CVPR, 2024, pp. 20 299–20 309.
[43] S. Giebenhain, T. Kirschstein, M. Georgopoulos, M. R¨unz, L. Agapito,
and M. Nießner, “Mononphm: Dynamic head reconstruction from
monocular videos,” in Proc. of CVPR, 2024, pp. 10 747–10 758.
[44] Y. Zhuang, B. Cheng, Y. Cheng, Y. Jin, R. Liu, C. Li, X. Cheng, J. Liao,
and J. Lin, “Learn2talk: 3d talking face learns from 2d talking face,”
IEEE Trans. Vis. Comput. Graph., vol. 31, no. 9, pp. 5829–5841, 2025.
[45] Y. Zhuang, C. Ma, Y. Cheng, X. Cheng, J. Liao, and J. Lin,
“Talkingeyes: Pluralistic speech-driven 3d eye gaze animation,” arXiv
preprint arXiv:2501.09921, 2025.
[46] S. Lin, L. Yang, I. Saleemi, and S. Sengupta, “Robust high-resolution
video matting with temporal guidance,” in Proc. of WACV, 2022, pp.
238–247.
[47] D. E. King, “Dlib-ml: A machine learning toolkit,” J. Mach. Learn. Res.,
vol. 10, pp. 1755–1758, 2009.
[48] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embed-
ding for face recognition and clustering,” in Proc. of CVPR, 2015, pp.
815–823.
[49] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular
margin loss for deep face recognition,” in Proc. of CVPR, 2019, pp.
4690–4699.
[50] A. R¨ossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and
M. Nießner, “Faceforensics++: Learning to detect manipulated facial
images,” in Proc. of ICCV, 2019, pp. 1–11.
[51] K. Shiohara, X. Yang, and T. Taketomi, “Blendface: Re-designing
identity encoders for face-swapping,” in Proc. of ICCV, 2023, pp. 7600–
7610.
[52] Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, “Accurate 3d
face reconstruction with weakly-supervised learning: From single image
to image set,” in Proc. of CVPRW, 2019, pp. 285–295.
[53] B. Doosti, S. Naha, M. Mirbagheri, and D. J. Crandall, “Hope-net: A
graph-based model for hand-object pose estimation,” in Proc. of CVPR,
2020, pp. 6607–6616.
[54] Z. Chen, J. He, X. Lin, Y. Qiao, and C. Dong, “Towards real-world
video face restoration: A new benchmark,” in Proc. of CVPR, 2024, pp.
5929–5939.
[55] F. Rosberg, E. E. Aksoy, F. Alonso-Fernandez, and C. Englund,
“Facedancer: Pose- and occlusion-aware high fidelity face swapping,”
in Proc. of WACV, 2023, pp. 3443–3452.
[56] Z. Liu, M. Li, Y. Zhang, C. Wang, Q. Zhang, J. Wang, and Y. Nie,
“Fine-grained face swapping via regional GAN inversion,” in Proc. of
CVPR, 2023, pp. 8578–8587.
[57] L. Xu, C. Lu, Y. Xu, and J. Jia, “Image smoothing via l0 gradient
minimization,” ACM Trans. Graph., vol. 30, no. 6, p. 174, 2011.
[58] X. Cheng, M. Zeng, and X. Liu, “Feature-preserving filtering with l0
gradient minimization,” Comput. Graph., vol. 38, pp. 150–157, 2014.
[59] X. Cheng, Y. Feng, M. Zeng, and X. Liu, “Video segmentation with l0
gradient minimization,” Comput. Graph., vol. 54, pp. 38–46, 2016.
[60] X. Cheng, M. Zeng, J. Lin, Z. Wu, and X. Liu, “Efficient l0 resampling
of point sets,” Comput. Aided Geom. Des., vol. 75, 2019.
