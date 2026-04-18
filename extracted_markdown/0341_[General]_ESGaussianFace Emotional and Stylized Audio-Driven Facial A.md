<!-- page 1 -->
1
ESGaussianFace: Emotional and Stylized
Audio-Driven Facial Animation via 3D Gaussian
Splatting
Chuhang Ma, Shuai Tan, Ye Pan∗, Jiaolong Yang and Xin Tong
Identity
Emotion
Style
Audio
Emotion
Style
Emotion
Style
Non-stylized
Stylized
Multi-view
"Angry"
Non-stylized
Stylized
Multi-view
"Happy"
Non-stylized
Stylized
Multi-view
"Surprised"
Emotion Control 
Art Style Control 
Multi-view Generation
Real-time Rendering
Fig. 1: We propose an Audio-Driven Facial Animation (ADFA) framework, named ESGaussianFace. This method is based on
3D Gaussian Splatting and supports the generation of talking heads with diverse emotions art styles. In contrast to prior ADFA
approaches, our framework enables the training of a single model capable of generating high-precision, multi-view consistent
videos with varying emotions in real time.
Abstract—Most current audio-driven facial animation research
primarily focuses on generating videos with neutral emotions.
While some studies have addressed the generation of facial videos
driven by emotional audio, efficiently generating high-quality
talking head videos that integrate both emotional expressions and
style features remains a significant challenge. In this paper, we
propose ESGaussianFace, an innovative framework for emotional
and stylized audio-driven facial animation. Our approach lever-
ages 3D Gaussian Splatting to reconstruct 3D scenes and render
videos, ensuring efficient generation of 3D consistent results.
We propose an emotion-audio-guided spatial attention method
that effectively integrates emotion features with audio content
features. Through emotion-guided attention, the model is able to
reconstruct facial details across different emotional states more
accurately. To achieve emotional and stylized deformations of the
3D Gaussian points through emotion and style features, we intro-
duce two 3D Gaussian deformation predictors. Futhermore, we
C. Ma, S. Tan and Y. Pan are with JHC & AI Institute, Shanghai Jiao
Tong University, Shanghai, China, E-mail:{mch3148300494, tanshuai0219,
whitneypanye}@sjtu.edu.cn. J. Yang and X. Tong are with Microsoft Research
Asia. Corresponding author: Ye Pan.
propose a multi-stage training strategy, enabling the step-by-step
learning of the character’s lip movements, emotional variations,
and style features. Our generated results exhibit high efficiency,
high quality, and 3D consistency. Extensive experimental results
demonstrate that our method outperforms existing state-of-the-
art techniques in terms of lip movement accuracy, expression
variation, and style feature expressiveness.
Index Terms—Facial animation, neural networks, 3D gaussian
splatting.
I. INTRODUCTION
A
UDIO-DRIVEN Facial Animation (ADFA) generates
facial animations for specific characters based on a given
audio segment. This task has widespread applications in digital
humans, virtual reality, and various other industrial domains.
Most existing ADFA methods primarily focus on neutral
emotions [1]–[3], with only a few [4]–[7] using audio with
different emotions to generate talking head videos. However,
due to the lack of constraints from a 3D scene, the videos
arXiv:2601.01847v1  [cs.CV]  5 Jan 2026

<!-- page 2 -->
2
produced by these methods lack 3D consistency and cannot
achieve multi-view rendering. In recent years, Neural Radiance
Fields (NeRF) [8] have achieved significant breakthroughs
in 3D reconstruction by modeling 3D scenes using implicit
functions. However, NeRF suffers from slow rendering speeds,
limiting its ability to achieve real-time rendering.
3D Gaussian Splatting (3DGS) [9] offers a promising solu-
tion to address this limitation. 3DGS is a 3D reconstruction
method that enables high-speed rendering through explicit
point-based 3D scene representation and a highly parallel
workflow, allowing for near-real-time rendering while main-
taining visual quality. Consequently, many ADFA approaches
[10], [11] have begun to use 3DGS to model heads of specific
characters. However, these efforts primarily focus on ADFA
with neutral emotion. Although EmoTalkingGaussian [12] is
capable of generating talking heads with predefined emotion
labels, generating ADFA videos that integrate 3D consistency,
emotional expression and style features from an emotional
image and a stylized avatar has not yet been explored.
We propose a novel framework named ESGaussianFace,
designed to generate both emotional and stylized talking head
videos efficiently. Fig. 2 illustrates the limitations of current
state-of-the-art ADFA methods in handling multi-emotion
ADFA tasks. Traditional ADFA methods [1], [2], depicted in
Fig. 2 (a), often suffer from a lack of 3D consistency, resulting
in inaccurate reconstructions of facial orientation and pose.
NeRF/3DGS-based ADFA methods [13], [14], shown in Fig.
2 (b), are proficient at capturing 3D features but struggle with
audios that contain mixed emotions. They typically require
training multiple models for different emotions, leading to
significant time and memory overhead. In contrast, our ES-
GaussianFace efficiently generates videos for a wide range
of emotions using a single model. Moreover, ESGaussianFace
allows for the seamless integration of any artistic avatar’s style
into emotional videos. During training, ESGaussianFace takes
driving audios, emotional videos, and style images as inputs.
These inputs provide the character’s lip movements, emotional
expressions, and style features, respectively. During inference,
any emotional video or image can be used as the emotion
source to extract emotion features.
To realize the aforementioned advantages, our ESGaus-
sianFace is structured into three modules: triplane-based 3D
Gaussian generator, audio-visual feature extraction and fusion
module, and ESGaussian deformation prediction module. The
triplane-based 3D Gaussian generator focuses on generating
the Gaussian parameters for the canonical face. We employ
a multi-resolution triplane to encode spatial information form
a standard 3D head, and a triplane decoder to generate the
canonical 3D Gaussian parameters.
Our goal is to train a 3D Gaussian model capable of
accurately generating the target avatar’s lip movements and
emotional expressions. The former is primarily driven by
the input audio, while the latter is controlled by a facial
image representing a specific emotion. To achieve this, we
extract content features from the audio and employ a 3D
Morphable Model [15] to capture the facial expression coef-
ficients as emotion features in audio-visual feature extraction
and fusion module. However, efficiently integrating these two
Emotional Audio
Emotional Video
(a) Traditional
ADFA Method
Inconsistency
Train Data
(b) NeRF/3DGS 
based ADFA
Method
···
High
Overhead
Fig. 2: (a) Traditional ADFA methods and (b) NeRF/3DGS-
based ADFA approaches struggle to handle multi-emotion
talking head generation tasks.
feature types to precisely control the 3D Gaussian deformation
presents a significant challenge. We hope that these two
features can dynamically influence distinct facial regions and
control the deformation of 3D Gaussian points within these
regions. For this purpose, we propose an emotion-audio-guided
spatial attention module, which consists of two primary cross-
attention layers. The audio-guided attention layer captures the
influence of the audio content features on the mouth region,
while the emotion-guided attention layer controls the deforma-
tion of various facial regions under the guidance of emotions.
The module leverages spatial attention to effectively integrate
content and emotion features, enabling precise guidance of the
3D Gaussian parameter deformations.
To further generate emotional and stylized talking head
videos, we incorporate the ESGaussian deformation prediction
module. We combine features output by the previous module
with the embeddings of 3D point positions to learn more
precise emotional variations. Futhermore, we extract style
encodings to capture the stylistic attributes of a specific artistic
avatar. These features guide the deformation predictor in gen-
erating Gaussian deformations. Training a model to directly
accomplish such a complex task is challenging. To overcome
this, we propose a multi-stage training strategy. First, the
neutral stage enables the model to predict lip movements under
neutral emotion. In the emotion stage, the model learns how

<!-- page 3 -->
3
different emotions influence the Gaussian parameters, based
on the learned lip movements. Finally, to achieve the stylized
deformation, we introduce a stylization stage.
Overall, ESGaussianFace can generate emotional and styl-
ized talking head videos, merging realism, accuracy, and
high efficiency. Experimental results show the superiority of
our method compared to state-of-the-art methods. The main
contributions of this work can be summarized as follows:
• We propose a novel framework for tackling the ADFA
task that combines both emotion and style features. To
the best of our knowledge, we are the first to achieve
this task while ensuring efficiency, accuracy, and multi-
view generation.
• We introduce an emotion-audio-guided spatial attention
method to accurately learn the influence of both the audio
and the emotion on dynamic spatial points. This method
enables the precise prediction of Gaussian deformations.
• We design a multi-stage training strategy, consisting of
three stages, to train the model. This strategy enables the
model to generate more accurate and stable videos. We
also introduce several novel loss functions for this task.
II. RELATED WORK
A. Audio-Driven Facial Animation in 2D Pixel Space
Audio-Driven Facial Animation (ADFA) tasks involve creat-
ing facial animation videos from an input audio clip. Early ap-
proaches primarily employ CNN-based encoder-decoder archi-
tectures [16], [17] or adversarial networks [2], [3], [18]–[24]
to generate talking head videos. However, they often neglect
facial structural features, leading to significant distortions in
the generated images. To address this issue, several techniques
[1], [25]–[27] enhance model accuracy by utilizing facial
landmarks as an additional control mechanism. Furthermore,
integrating the 3D Morphable Model (3DMM) [15], [28]–[33]
and 3D blendshape face model [34]–[40] into ADFA system-
sproduces more realistic and accurate videos. However, these
works primarily focus on faces displaying neutral emotions.
Currently, few studies address the generation of talking
head videos with varying facial emotions. MEAD [41] and
RAVDESS [42] datasets provide valuable emotional audio-
visual data with high quality. Several efforts [4]–[7], [43]–[65]
have developed methods for generating emotional talking head
videos. These methods derive emotion features from diverse
sources. For instance, EAT [7] utilizes discrete emotion labels,
while DreamTalk [6] relies on emotional images. However,
they often fall short in managing facial aspects such as orien-
tation and pose, and they cannot render multi-view videos.
B. Audio-Driven Facial Animation via Neural Rendering
Recently, Neural Radiance Fields (NeRF) [8] have shown
exceptional performance in rendering complex 3D scenes. As a
result, many ADFA methods [13], [66]–[72] have incorporated
NeRF into their frameworks. For instance, AD-NeRF [13]
innovatively separates facial generation into head and torso
stages, leveraging NeRF’s implicit representation to achieve
high-quality rendering. However, a major limitation of NeRF
is its slow rendering speed, presenting a challenge in balancing
efficiency and accuracy in ADFA.
The advent of 3D Gaussian Splatting (3DGS) [9] has
significantly addressed this issue by offering both improved
rendering quality and faster speeds compared to NeRF. Recent
studies in 3D facial animation have begun to adopt 3DGS for
its high precision and efficiency. HeadGas [10] is pioneering
in applying 3DGS into 3D facial reconstruction. MonoGaus-
sianAvatar [11] utilizes linear blend skinning to map the 3D
points from the canonical space to the deformed space, en-
abling effective ADFA. Currently, numerous studies [73]–[77]
integrate the FLAME model [78] and parameterized tri-plane
[79] into 3DGS methods, yielding more accurate outcomes.
For instance, GaussianTalker [14] employs a multi-resolution
tri-plane to represent canonical facial shapes and uses a spatial-
audio attention module to predict 3D Gaussian deformation.
EmoTalkingGaussian [12] utilizes predefined emotion labels
and predicts Gaussian parameter deformations to infuse emo-
tions into talking heads. However, no existing method has yet
explored facial stylization based on 3DGS.
C. Style Transfer on Facial Images and Videos
Facial style transfer is a well-explored research area. Du-
alStyleGAN [80] employs the pSp encoder [81] to extract
style features from any style image and integrates them with
structural features from facial images. Utilizing StyleGAN
[82], this approach generates high-quality stylized images
based on the combined features. VToonify [83] extends this
capability to video, enabling the creation of high-precision
stylized videos. In ADFA domain, Style2Talker [84] extracts
style features from style images and controls emotions us-
ing a diffusion model. However, it is constrained by its
2D generation framework. Although a few studies [85]–[87]
have explored 3D facial style transfer, no existing work is
capable of achieving 3D facial animation while simultaneously
enabling control over arbitrary emotion and style. In contrast,
our method utilizes 3DGS to achieve ADFA with various
emotional expressions and styles efficiently.
III. PRELIMINARY: 3D GAUSSIAN SPLATTING
3D Gaussian Splatting (3DGS) [9] can learn an explicit
3D representation from given images and camera parameters,
enabling the reconstruction of stable scenes by a series of
3D Gaussian splats. Typically, a Gaussian splat is defined by
its center position µ ∈R3, scaling factor s ∈R3, rotation
quaternion q ∈R4, k-degree spherical harmonics coefficients
sh ∈R3(k+1)2, and opacity value α ∈R. Therefore, a
Gaussian splat can be described as G = {µ, s, q, sh, α}. Each
3D Gaussian can be computed as follows:
g(x) = e−1
2 (x−µ)T Σ−1(x−µ).
(1)
The semi-definite covariance matrix Σ can be computed from
a scaling matrix S and a rotation matrix R, defined by s and
r, respectively:
Σ = RSST RT .
(2)
During the rendering process, 3D Gaussians need to be
projected onto the 2D image plane within a specific camera

<!-- page 4 -->
4
(Emotional Part            and Stylized Part           )
Triplane-based 3D
Gaussian Geneator (        )
*Emotional Audio
*Emotion Source
AGA
C
*Style Source
Triplane
C
Pretrained Model
C
ESGaussian Deformation
Prediction Module
Audio-visual Feature Extraction
and Fusion Module (        )
Feature Concat
EGA
FF
*
Model Input
ESAM
*
*
*
Multi-stage Training Strategy
Train
Freeze
Fig. 3: The overview of our proposed ESGaussianFace model. During inference, we initialize the Gaussian parameters of the
canonical face with the Triplane-based 3D Gaussian Generator (Sec. IV-A). The Audio-Visual Feature Extraction and Fusion
Module (Sec. IV-B) extracts content and emotion features from the audio and emotional source separately, then combines them
through the Emotion-audio-guided Spatial Attention Module (ESAM). The resulting fused features are subsequently input
into the ESGaussian Deformation Prediction Module (Sec. IV-C), which predicts the emotional and stylized deformations of
the Gaussian parameters. For training, a Multi-stage Training Strategy (Sec. IV-D) is adopted, with the model learning lip
movements, emotional expressions, and style features in three stages.
coordinate system. The covariance matrix Σ′ ∈R2 in 2D
space can be obtained from the view transformation matrix
W and the Jacobian matrix J of the approximated projection
transformation [88]:
Σ′ = JWΣWT JT .
(3)
The 3D Gaussians associated with each pixel can be sorted
based on their respective depths. The pixel color C is com-
puted by blending all N Gaussians in the depth-sorted order:
C =
N
X
i=1
ciα′
i
i−1
Y
j=1
(1 −α′
j),
(4)
where ci is the color derived from the spherical harmonics
coefficients of the i-th Gaussian with view direction, and
α′
i denotes the opacity obtained by the multiplication of the
opacity of the i-th Gaussian with the covariance matrix Σ′.
IV. METHOD
The proposed ESGaussianFace framework is shown in
Fig. 3. It mainly comprises three parts: a triplane-based
3D Gaussian generator (Sec. IV-A), an audio-visual feature
extraction and fusion module (Sec. IV-B) and an ESGaussian
deformation prediction module (Sec. IV-C).
The triplane-based 3D Gaussian generator decodes features
generated by a triplane to obtain the Gaussian parameters of
the canonical face. In the audio-visual feature extraction and
fusion module , we extract both audio and emotion features
from the input to guide the computation of the spatial atten-
tion. The ESGaussian deformation prediction module predicts
the deformation of the Gaussian parameters under emotional
and stylistic control, and uses a neural renderer to generate
emotional and stylized talking head videos. Additionally, we
introduce a multi-stage training strategy (Sec. IV-D) designed
to enable the model to effectively capture and predict defor-
mations across a diverse range of emotions and styles.
A. Triplane-based 3D Gaussian Generator
In this section, we introduce the implementation specifics
of the triplane-based 3D Gaussian generator. To achieve high-
quality and 3D-consistent facial animation results, we employ
3DGS to obtain explicit 3D representations. We initialize N
sets of 3D Gaussians based on a 3DMM model of the standard
human face. Subsequently, we encode the positional features
of all 3D Gaussians using a multi-resolution tri-plane [71]. The
tri-plane Tc consists of three axis-aligned orthogonal feature
planes. For any 3D position µ ∈R3, we project it onto Tc to
obtain the corresponding feature vector fp:
fp = (Fxy, Fyz, Fxz) = Tc(µ).
(5)
Each plane F has a resolution of R × R × C, where R
denotes the spatial resolution and C represents the number of
channels. fp is then fed into a tri-plane decoder Dc, enabling
the extraction of the canonical Gaussian parameters Gcan.

<!-- page 5 -->
5
(a) Rendered
(b) Audio
(c) Emotion
(d) Eye
(e) Viewpoint
(f) Null
Fig. 4: (a) Shows the rendered image, while the other panels
display the attention score distributions for (b) audio content,
(c) emotions, (d) eye blinks, (e) head orientations, and (f)
temporal consistency.
B. Audio-visual Feature Extraction and Fusion Module
This module performs the extraction of audio and video
features, as well as the feature fusion via spatial attention.
1) Audio-visual Feature Extraction: We aim to extract
content features from the driving audio to guide the generation
of lip movements in the human face. For an emotional audio
segment A1:T , we extract its Deep Speech features [89]
a1:T , which effectively capture the content of the audio. To
capture contextual information, we extend at ∈R16×29 by l
frames both forward and backward, resulting in at−l:t+l with
a temporal length of 2l.
To imbue the generated results with different emotions, we
use an emotional video (or an emotional image) as the emotion
source. For emotion feature extraction, we employ a pretrained
Deep3D [90] model as the 3D facial extractor Ed. This model,
based on 3DMM, performs 3D reconstruction and extracts 3D
coefficients of 2D facial images. Here we use the expression
coefficients e
∈
R64 as the emotion feature. Following
previous works, we extract the AU45 feature y ∈R [91]
from Iemo to quantify the degree of blinking. Furthermore,
we capture the extrinsic camera pose of the video and encode
the viewpoint feature v ∈R12.
2) Emotion-audio-guided Spatial Attention: We aim for
the audio and emotion features to influence different facial
regions and guide the deformation of Gaussian points within
them. For this purpose, we introduce an emotion-audio-guided
spatial attention method based on spatial-audio attention [14].
The spatial attention dynamically allocate weights to different
regions in space, enabling the fusion of audio and emotion fea-
tures while controlling the 3D Gaussian points. We integrates
B Emotion-audio-guided Spatial Attention Modules (ESAMs).
Each ESAM is composed of two cross-attention layers and
a Feed-forward (FF) layer. The first, known as the Audio-
guided Attention (AGA) layer, forecasts the effect of audio
on distinct facial regions. In contrast, the second layer, the
Emotion-guided Attention (EGA) layer, estimates the influence
of different emotions across various facial regions:
z0 = fp,
(6)
z′
b = Fca(zb−1, f t
c) + zb−1,
b = 1, ..., B,
(7)
z′′
b = Feg(z′
b, f t
e) + z′
b,
b = 1, ..., B,
(8)
zb = Ffl(z′′
b ) + z′′
b ,
b = 1, ..., B,
(9)
where f t
a = {at−l:t+l, y, v, ∅}, f t
e = {e, ∅}. ∅is an empty
vector, ensuring the consistency of global features across
different frames. Fag, Feg and Ffl represent AGA, EGA and
FF, respectively.
We visualize the attention scores of different features across
various facial regions, as shown in Fig. 4. It is evident that
the audio features primarily affect areas around the mouth,
while emotion features influence multiple regions that are
more sensitive to emotional variations, such as the eyes,
eyebrows and mouth. This significantly ensures the stability
and accuracy of video generation in dynamic scenarios with
changes of audio contents and emotional expressions.
C. ESGaussian Deformation Prediction Module
To obtain the emotional and stylized ADFA results, we
introduce the ESGaussian deformation prediction module.
This module consists of two Gaussian parameter deformation
predictors, Pemo and Psty. In addition to the spatially-aware
feature zB, we incorporate the emotion feature e to further
capture different emotions. We also discover that incorporating
the positional embedding of 3D points yields more precise
results. For this, we use an MLP network Eµ as encoder to
obtain the encoded feature p ∈R64 of the 3D positions.
We input the feature femo = {zB, e, p} into emotional
deformation predictor Pemo to obtain the Gaussian deforma-
tion ∆Gemo = {∆µemo, ∆semo, ∆qemo, ∆shemo, ∆αemo}.
Adding ∆Gemo to Gcan yields Gemo, which are input to the
3DGS neural renderer R to generate the image ˆIemo that
reflects the emotional expression:
ˆIemo = R(Gemo) = R(Gcan + Pemo(femo)).
(10)
Regarding the stylized deformation predictor Fsty, it also
utilizes the spatially-aware feature zB, the emotion feature e
and the position feature p as inputs. Furthermore, we utilize a
pretrained style encoder [81] and an MLP encoder jointly as
style extractor Es to capture extrinsic style feature s ∈R128
of the style images. The feature fsty = {zB, e, p, s} is then
input into Psty to obtain the Gaussian deformation ∆Gsty.
This deformation imparts the style of Is to the emotional 3D
Gaussians, thereby rendering images ˆIsty that combine both
the art style and emotional expression.
D. Multi-stage Training Strategy
During training, the Gaussian deformation predictor often
fails to directly learn the deformations from a neutral face
to faces with lip movements and different emotions. Directly
training the model leads to weak emotional expressions, loss of
facial features, or even distortion. To address this, we propose
a multi-stage training strategy to obtain more accurate results.
1) Neutral Stage with Lip Movement: First, we train the
triplane-based 3D Gaussian generator MG and the emotional
deformation prediction module M emo
P
(Eµ and Pemo). The
goal of this stage is to enable the MG to accurately generate
the Gaussian parameters Gcan of a canonical face, while
allowing M emo
P
to learn the lip movement of neutral emotion.
We use the t-th frame of the talking head video I1:T
neu, which
depicts neutral emotion, for training supervision. We employ
a loss function that comprises six distinct components:
L = λ1Lrgb + λ2Lper + λ3Lssim
+ λ4Llip + λ5Lld + λ6Lsmo,
(11)

<!-- page 6 -->
6
TABLE I: Quantitative comparisons of stylized emotional ADFA results with state-of-the-art methods. The top-performing
results are highlighted in bold, while the second-best results are underlined. For each method, We also provide its additional
capabilities in generating emotional expression, art style, and multi-view images.
Method
MEAD [41]
RAVDESS [42]
Output
PSNR↑
SSIM↑
FID↓
LPIPS↓
Sync↑
LMD↓
Accemo↑
PSNR↑
SSIM↑
FID↓
LPIPS↓
Sync↑
LMD↓
Accemo↑
Emotion
Style
Multi-view
MakeItTalk [1]
27.597
0.602
53.090
0.059
4.495
5.208
19.379
27.760
0.689
29.168
0.058
5.135
5.786
20.468
%
%
%
Wav2Lip [2]
27.819
0.670
46.373
0.051
5.851
5.068
16.802
27.931
0.715
32.567
0.057
6.169
5.623
18.027
%
%
%
Audio2Head [3]
26.529
0.623
63.287
0.070
3.342
6.681
19.915
25.639
0.583
69.613
0.069
2.837
7.243
19.322
%
%
%
MEAD [41]
28.081
0.712
21.273
0.039
5.784
3.564
64.920
–
–
–
–
–
–
–
!
%
%
EAMM [4]
26.807
0.681
58.334
0.063
3.457
4.578
25.415
26.071
0.643
50.811
0.068
4.106
5.746
27.133
!
%
%
EAT [7]
27.650
0.690
35.094
0.049
5.277
4.360
58.485
26.134
0.613
40.736
0.067
3.662
6.459
53.869
!
%
%
DreamTalk [6]
27.801
0.732
34.239
0.051
5.560
4.693
61.810
26.193
0.650
38.826
0.064
5.076
6.118
57.090
!
%
%
EDTalk [52]
27.938
0.760
34.497
0.048
5.436
4.467
62.288
26.466
0.667
36.221
0.060
4.720
5.854
59.315
!
%
%
Style2Talker [84]
29.455
0.795
25.282
0.035
5.294
3.007
70.273
26.906
0.647
41.649
0.061
3.410
6.186
49.544
!
!
%
AD-NeRF [13]
25.282
0.549
84.506
0.086
1.298
–
10.245
25.021
0.508
86.448
0.082
1.132
–
10.714
%
%
!
SyncTalk [92]
28.120
0.803
32.959
0.040
4.770
5.071
48.245
27.706
0.697
31.793
0.053
3.857
5.646
47.680
%
%
!
GaussianTalker [14]
28.911
0.818
22.290
0.038
5.202
4.981
52.621
28.516
0.747
25.260
0.044
4.354
5.329
51.319
%
%
!
ESGaussianFace
31.873
0.901
16.933
0.028
6.216
2.832
75.944
30.772
0.833
21.796
0.034
5.757
3.145
71.124
!
!
!
where Lrgb, Lper, and Lssim represent the L1 color loss,
perceptual loss [93], and SSIM loss [94], respectively; λi
denotes the weighting factor of the loss function. To ensure
consistency of lip movements with the audio, we employ L1
loss to the mouth region to accurately align the lip movements:
Llip = ∥mneu · Ineu −ˆmneu · ˆIneu∥1,
(12)
where mneu denotes the mask of the mouth region extracted
by a face parsing model [95]. Furthermore, we utilize the
facial landmarks to ensure the accuracy of facial structure and
emotions. This leads to the landmark distance loss Lld:
Lld = ∥w(Fld(Ineu) −Fld(ˆIneu))∥2
2,
(13)
Fld denotes the pretrained landmark detector, and wi ∈w is
the weight for the i-th landmark. Moreover, we add a smooth
loss Lsmo to eliminate temporal jitter in generated videos:
Lsmo = ∥∆Gt
neu −0.5 × (∆Gt−1
neu + ∆Gt+1
neu)∥2
2.
(14)
In this stage, we select talking head videos exhibiting neutral
emotion of a specific actor from the emotional dataset. The
objective of this stage is to enable MG to learn the person’s
explicit 3D representation, while allowing MF and M emo
P
to
learn lip movements under varying audio conditions.
2) Emotion Stage with Emotional Deformation: Building
on the pretraining of the model, the goal of this stage is to
train the entire network (excluding the stylization component).
Here, we fix the weights of MG. After pretraining, M emo
P
is capable of predicting lip movements of neutral emotion.
At this point, the model only needs to predict the emotional
deformation of the Gaussian parameters with the same lip
movement. We supervise the model’s generated ˆIemo using
the same loss function as in the neutral stage for training.
Since the model has already learned the 3D facial repre-
sentation of the current actor, we fix the parameters of MG in
emotion stage. We select talking head videos of the same actor
and the same audio but with different emotions for training.
As MF and M emo
P
are already capable of predicting lip
movements corresponding to a given audio, this stage simply
continues training these two modules based on the parameters
obtained in the neutral stage, enabling them to learn the overall
facial emotional deformation under consistent lip movements.
3) Stylization Stage with 3DGS Style Transfer: To further
train the stylization functionality, we introduce a third training
stage. We employ the VToonify [83] model on the emotional
video I1:T
emo used in emotion stage and the style image Is
to achieve stylization of emotional facial video, resulting in
I1:T
sty . In this stage, M sty
P
(Es and Psty) aims to learn the
stylized deformation of the 3D Gaussian parameters guided by
the style features s. In addition to the loss functions from L,
we introduce a extrinsic style loss Lsty to ensure consistency
between the generated results and the style of Is:
Lexs = ∥Es(Isty) −Es(ˆIsty)∥1.
(15)
After completing the multi-stage training, we obtain a
person-specific 3D Gaussian model capable of controlling
arbitrary emotions and styles. During inference, since MG has
already learned the 3D Gaussian representation of the target
person, no additional input image of the individual is required.
We only need to provide the target driving audio, an image
representing the target emotion (emotion source), and an image
with the desired art style (style source). ESGaussianFace can
then generate a 3D talking head with accurate lip synchroniza-
tion, realistic emotional expression, and faithful stylization.
V. EXPERIMENTS
A. Experimental Settings
1) Datasets: In our experiments, we use the MEAD [41]
and RAVDESS [42] datasets for training. MEAD is a high-
quality publicly available dataset that contains audio-visual
data of 8 emotions performed by 60 actors. To demonstrate
the generalizability of our model across different datasets,
we also incorporate the RAVDESS dataset, which consists
of audio-visual data representing 8 emotions from 24 actors.
Additionally, we use various art datasets [96] to obtain art style
references. For further details on the datasets and experimental
parameters, please consult the supplementary materials.
2) Comparison Setting: To perform a comprehensive com-
parison of talking head videos with both emotional expres-
sions and art styles, we follow the experimental approach of
Style2Talker [84]. We input the source facial image and the
style image into VToonify [83] to generate a stylized facial im-
age. This image is then processed using several state-of-the-art

<!-- page 7 -->
7
Emotion
Source
(Sad      )
MakeItTalk
Wav2Lip
Audio2Head
EAMM
GaussianTalker
Ground Truth
Ours
AD-NeRF
Source
Avatar
Emotion
Source
(Surprise      )
Source
Avatar
Style
Source
Style
Source
Source
Avatar
Style
Source
Source
Avatar
Style
Source
Emotion
Source
(Happy      )
Emotion
Source
(Angry      )
(a)
(b)
(c)
(d)
MEAD
EAT
ED-Talk
SyncTalk
DreamTalk
Fig. 5: Qualitative comparisons of stylized emotional ADFA results with state-of-the-art methods. Experiments (a) and (b)
present results on the RAVDESS and MEAD datasets, respectively, where the source avatar and emotion source are derived
from the same dataset in each experiment. Experiments (c) and (d) show results with the emotion source from different datasets.
ADFA methods, achieving results comparable to our method.
We select three traditional ADFA methods: MakeItTalk [1],
Wav2Lip [2], and Audio2Head [3]. In the category of ADFA
methods based on NeRF or 3DGS, we choose AD-NeRF
[13], SyncTalk [92] GaussianTalker [14]. For emotional ADFA
methods, we include MEAD [41], EAMM [4], EAT [7],
DreamTalk [6] and EDTalk [52] in our comparisons.
We employ PSNR, SSIM [94], FID [97], and LPIPS [98]
to evaluate the quality of the generated images and their
similarity to real images. To compare the consistency of lip
movements with the audio, we use the confidence score of
SyncNet [99] (Sync). Additionally, we measure the accuuracy
of expressions and poses by the average distance between
landmarks [100] of the generated and real faces (LMD), and
further assess emotional accuracy using Accemo [101].
B. Experimental Results
1) Quantitative Results: The quantitative comparison of the
stylized emotional ADFA results between our method and
state-of-the-art methods is given in Tab. I. As observed, we
are the only method that supports generation of emotional ex-
pressions, art styles, and multi-view rendering. It consistently
outperforms others on most evaluation metrics, ranking second
only to Wav2Lip in Sync. This is attributed to Wav2Lip’s
use of SyncNet as discriminator during training. Notably, the
lowest LMD demonstrates that our method is the most accurate
in representing lip movements and emotional expressions.
2) Qualitative Results: Fig. 5 presents a qualitative com-
parisons of the results. Experiments (a) and (b) present test
results on the RAVDESS and MEAD datasets, respectively,
where the source avatar and emotion source are derived from
the same dataset in each experiment. Experiments (c) and (d)
show results with the emotion source selected from different
datasets. For stylization, experiments (a) and (b) show the
results without color transfer, while experiments (c) and (d) in-
clude color transfer. As observed, while GaussianTalker excels
at restoring facial poses and details, it tend to confuse different
emotions (circled in yellow boxes). Compared to other meth-
ods, our results demonstrate significantly improved accuracy
in lip movements (indicated by blue boxes). Furthermore, our
method supports multi-view video generation, showcasing its
versatility and effectiveness. More experimental results and an
in-depth analysis are provided in the supplementary material.
3) Inference Speed: We evaluate the efficiency using FPS.
Our method achieves an FPS of 69, enabling real-time ren-
dering and generation. As shown in Tab. II, our method out-
performs most state-of-the-art methods in terms of efficiency,
ranks second only to GaussianTalker. This is attributed to the
incorporation of additional emotion and stylization modules.
Nevertheless, the lightweight predictor in our model enables
flexible control over arbitrary emotions and styles, resulting in
only a minimal FPS difference compared to GaussianTalker.
4) Emotion and Style Manipulation: We can achieve emo-
tion and style manipulation through linear interpolation:
f = αf1 + (1 −α)f2.
(16)
Here, f denotes the emotion or style feature, and α represents
the interpolation weight. For emotion manipulation, we use the

<!-- page 8 -->
8
TABLE II: We use FPS to measure the efficiency of reenactment. The top-performing results are highlighted in bold, while
the second-best results are underlined.
Method
MakeItTalk
Wav2Lip
Audio2Head
MEAD
EAMM
EAT
DreamTalk
EDTalk
Style2Talker
AD-NeRF
SyncTalk
GaussianTalker
ESGaussianFace
FPS
14.290
15.243
13.817
5.715
8.351
15.360
7.832
16.878
14.905
0.112
1.030
76.802
69.624
Style 1
Identity
Style 2
Emotion 1
(Happy)
Emotion 2
(Angry)
Fig. 6: Emotion and style manipulation results. We set the
interpolation weight to 1, 0.75, 0.5, 0.25, and 0, respectively.
emotion feature e encoded by 3D facial extractor Ed. For style
manipulation, we use the style feature s extracted from the
style source Is. The results of emotion and style manipulation
under different interpolation weights are shown in Fig. 6. We
set the value of α to 1, 0.75, 0.5, 0.25, and 0, respectively. As
shown in the figure, our method successfully achieves smooth
and continuous transitions in both emotion and style domains
by interpolating the corresponding features.
5) Multi-view Rendering Results:
Our method enables
multi-view rendering based on 3D Gaussian Splatting. We
present rendering results from various viewpoints, as shown
in Fig. 7. We select in-the-wild images representing angry,
surprised, happy and sad emotions as emotion sources. The
results demonstrate that our method can accurately control
both the target emotion and style, while maintaining strong
3D consistency across different rotation angles.
6) Generalization on Other Datasets: Most videos outside
of the emotional datasets [41], [42] exhibit neutral emotions,
making it challenging to find emotional videos that meet
the requirements of our training strategy. To demonstrate the
generalization capability of our method, we utilize EmoStyle
[103] to edit the emotion of each frame of any in-the-wild
talking head video. These emotion-edited videos are then used
to train our method. In Fig. 8, we present inference results on
videos beyond MEAD and RAVDESS datasets. These results
demonstrate the strong generalization ability of our method.
7) User Study: We conduct a user study to compare the
performance with state-of-the-art methods. We recruit 24 par-
ticipants (12 males and 12 females) and each is presented with
Source Avatar
Emotion (Angry)
Style
Input
Source Avatar
Emotion (Fear)
Style
Source Avatar
Emotion (Happy)
Style
Input
Multi-view
Results
Source Avatar
Emotion (Sad)
Style
Multi-view
Results
Fig. 7: Multi-view rendering results. We select in-the-wild
emotional images as emotion sources.
Source Avatar
Emotion (Angry)
Style (w/ CS)
Input
Results
Source Avatar
Emotion (Happy)
Style (w/ CS)
Source Avatar
Emotion (Sad)
Style (w/ CS)
Input
Results
Source Avatar
Emotion (Surprise) Style (w/o CS)
Fig. 8: Our method demonstrates strong generalization capa-
bilities to videos from other datasets [13], [102].
7 videos generated by 11 methods. Since MEAD can only
generate driving results for a specific avatar and the outputs of
AD-NeRF are blurry, we do not include them in the user study.
Participants are asked to rate the accuracy of lip movement,
emotional expressions, and art style in the generated videos,
using a scale from 1 to 11 for different methods. We then
identify the top-performing method for each participant and
illustrate their preferences using a pie chart, which is included
in Fig. 9 (a). Additionally, we calculate the average scores for

<!-- page 9 -->
9
MakeItTalk
Wav2Lip
Audio2Head
EAMM
ED-Talk
Lip Sync
Emotion
Art Style
EAT
SyncTalk
(a) Preference
(b) Score
GaussianTalker
ESGaussianFace
Style Talker
DreamTalk
Fig. 9: User Study Results. (a) The most preferred method of
each participant; (b) The score ranges from 1 to 11, and error
bars imply the standard deviations.
TABLE III: Quantitative Results for ablation study.
Method
PSNR ↑
SSIM ↑
FID ↓
LPIPS ↓
Sync ↑
LMD ↓
Accemo ↑
w/o ESAM
28.679
0.769
25.403
0.040
4.876
4.982
29.807
w/o MTS
28.050
0.748
30.672
0.046
2.154
4.781
36.263
w/o p
28.780
0.813
24.012
0.033
4.169
4.026
63.558
w/o e
29.455
0.814
23.124
0.036
5.061
4.816
27.921
w/o Llip
30.025
0.878
18.539
0.029
3.064
3.524
66.542
w/o Lld
30.842
0.823
22.094
0.033
4.440
4.776
53.540
w/o Lexs
29.189
0.790
26.303
0.038
5.435
3.022
71.251
Full Model
31.873
0.901
16.933
0.028
6.216
2.832
75.944
each method. The results for the three metrics are presented
in Fig. 9 (b). Our method demonstrate superior performance
in terms of lip movements, emotion and art style.
8) Ablation Study: To better demonstrate the effectiveness
of our method’s components, we conduct ablation experi-
ments. The quantitative results are presented in Tab. III, while
the qualitative results are shown in Fig. 10.
We design 7 variants in total: (a) First, we examine the con-
tribution of the emotion-audio-guided spatial attention module
introduced in Sec. IV-B (w/o ESAM). We replace our proposed
ESAM module with the spatial-audio attention module em-
ployed in GaussianTalker. As shown in Fig. 10 (a), the model’s
generated results, without ESAM, exhibit inaccurate emotional
expressions. Furthermore, the lower Accemo observed in the
quantitative results also indicates a decline in the accuracy of
emotion reconstruction in the absence of ESAM module. The
result highlights the crucial role of this module in controlling
the character’s emotional expression. (b) Next, we investigate
the necessity of the multi-stage training strategy proposed in
Sec. IV-D (w/o MTS). We bypass the first two stages and
directly train the full model using stylized emotional videos
as supervision. As seen in the results depicted in Fig. 10
(b), the output only shows slight color changes and does
not achieve the desired stylization effect. Additionally, the
emotional variation in the generated results is minimal, and
the accuracy of lip movements shows a slight decline. These
Source Avatar
(a) w/o ESAM
(b) w/o MTS
(e) w/o
Emotion Source
Style Source
(c) w/o
(d) w/o
(f) w/o
(g) w/o
Full Model
Ground Truth
Fig. 10: Qualitative ablation study results. The figure presents
an example of ADFA results depicting a happy emotion.
experimental findings suggest that directly training the full
model for such a complex task is inadequate. A multi-stage
approach, where lip movements, emotional expressions, and
style features are learned incrementally, is essential.
We also evaluate the necessity of two features, e and p,
used in the ESGaussian deformation prediction module. (c)
Although the expression feature e is incorporated during the
extraction of zB, it remains essential during the Gaussian
parameter prediction process. In ESAM, e primarily assigns
spatial attention weights across different facial regions. In
contrast, e provides emotion features to guide the emotion-
aware deformation of Gaussian parameters in the deformation
prediction module. Fig. 10 (c), LMD and Accemo in Tab.
III show that omitting e during the prediction process leads
to inaccurate emotions. (d) 3D Gaussian points at different
locations undergo varying degrees of deformation. Positional
embedding feature p provides spatial location priors, im-
proving the network’s understanding of emotion’s impact on
different facial regions. The visualized results of Fig. 10 (d)
and the PSNR values in Tab. III show that removing p leads
to a noticeable decline in both image quality and accuracy.
Furthermore, We analyze the loss functions proposed in our
method: (e) w/o Llip: the absence of this loss primarily leads
to a decrease in lip synchronization; (f) w/o Lld: the lack
of landmark constraints results in declines in various metrics
of the results; (g) w/o Lexs: the absence of style loss does
not significantly impact lip movement and landmark positions
but greatly reduces the accuracy of the generated images.
Both quantitative and qualitative results demonstrate that our
proposed loss functions are essential for training our model.
VI. LIMITATION
Despite the success of our work, we also recognize some
limitations. (a) First, since the MEAD dataset contains only 3
intensity levels for each emotion, our method tends to produce
similar results when processing emotion sources with subtle
differences. (b) Our method uses stylized videos processed by

<!-- page 10 -->
10
VToonify as ground truth. However, more advanced stylization
methods based on diffusion models have recently emerged.
(c) Our method can only train a 3D Gaussian field for a
specific character. To drive the talking head videos of different
characters, separate 3D Gaussian models need to be trained
for each. This is a common challenge faced by current ADFA
methods based on NeRF and 3D Gaussians. In future work,
we will focus on addressing these issues.
VII. CONCLUSION
In this paper, we introduce ESGaussianFace, a novel frame-
work for generating high-quality talking head videos that
incorporate both emotional expressions and style features.
Leveraging 3D Gaussian Splatting, our method ensures high
efficiency, 3D consistency, and multi-view rendering capa-
bilities. We design an emotion-audio-guided spatial attention
module that captures the influence of audio and emotion on
the position of Gaussian points. The features output by this
module, together with the encoded expression features and the
embedding of 3D point positions, ensure the precision of the
generated facial structures. Furthermore, to tackle this complex
task, we propose a multi-stage training strategy, where the
model learns the lip movements, emotional deformations, and
style features in three stages. Both qualitative and quantitative
results on multiple datasets demonstrate the superior perfor-
mance of our approach over existing state-of-the-art methods.
ACKNOWLEDGMENTS
This work was completed during a visit to MSRA under
the StarTrack program, hosted by Jiaolong Yang and Xin
Tong. This work was supported in part by National Natural
Science Foundation of China (NSFC, No. 62472285 and No.
62102255).
REFERENCES
[1] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis,
and D. Li, “Makelttalk: speaker-aware talking-head animation,” ACM
Transactions On Graphics (TOG), vol. 39, no. 6, pp. 1–15, 2020.
[2] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar, “A
lip sync expert is all you need for speech to lip generation in the
wild,” in Proceedings of the 28th ACM international conference on
multimedia, 2020, pp. 484–492.
[3] S. Wang, L. Li, Y. Ding, C. Fan, and X. Yu, “Audio2head: Audio-
driven one-shot talking-head generation with natural head motion,”
arXiv preprint arXiv:2107.09293, 2021.
[4] X. Ji, H. Zhou, K. Wang, Q. Wu, W. Wu, F. Xu, and X. Cao, “Eamm:
One-shot emotional talking face via audio-based emotion-aware motion
model,” in ACM SIGGRAPH 2022 Conference Proceedings, 2022, pp.
1–10.
[5] S. Tan, B. Ji, and Y. Pan, “Emmn: Emotional motion memory network
for audio-driven emotional talking face generation,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
22 146–22 156.
[6] Y. Ma, S. Zhang, J. Wang, X. Wang, Y. Zhang, and Z. Deng,
“Dreamtalk: When emotional talking head generation meets diffusion
probabilistic models,” arXiv preprint arXiv:2312.09767, 2023.
[7] Y. Gan, Z. Yang, X. Yue, L. Sun, and Y. Yang, “Efficient emotional
adaptation for audio-driven talking-head generation,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 22 634–22 645.
[8] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp.
99–106, 2021.
[9] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[10] H. Dhamo, Y. Nie, A. Moreau, J. Song, R. Shaw, Y. Zhou, and E. P´erez-
Pellitero, “Headgas: Real-time animatable head avatars via 3d gaussian
splatting,” arXiv preprint arXiv:2312.02902, 2023.
[11] Y. Chen, L. Wang, Q. Li, H. Xiao, S. Zhang, H. Yao, and Y. Liu,
“Monogaussianavatar: Monocular gaussian point-based head avatar,”
arXiv preprint arXiv:2312.04558, 2023.
[12] J. Cha, S. Yoon, V. Strizhkova, F. Bremond, and S. Baek, “Emotalk-
inggaussian: Continuous emotion-conditioned talking head synthesis,”
arXiv preprint arXiv:2502.00654, 2025.
[13] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, “Ad-
nerf: Audio driven neural radiance fields for talking head synthesis,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2021, pp. 5784–5794.
[14] K. Cho, J. Lee, H. Yoon, Y. Hong, J. Ko, S. Ahn, and S. Kim,
“Gaussiantalker: Real-time high-fidelity talking head synthesis with
audio-driven 3d gaussian splatting,” arXiv preprint arXiv:2404.16012,
2024.
[15] V. Blanz and T. Vetter, “A morphable model for the synthesis of 3d
faces,” in Seminal Graphics Papers: Pushing the Boundaries, Volume
2, 2023, pp. 157–164.
[16] R. Kumar, J. Sotelo, K. Kumar, A. De Brebisson, and Y. Ben-
gio, “Obamanet: Photo-realistic lip-sync from text,” arXiv preprint
arXiv:1801.01442, 2017.
[17] A. Jamaludin, J. S. Chung, and A. Zisserman, “You said that?: Syn-
thesising talking faces from audio,” International Journal of Computer
Vision, vol. 127, pp. 1767–1779, 2019.
[18] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,”
Advances in neural information processing systems, vol. 27, 2014.
[19] L. Chen, R. K. Maddox, Z. Duan, and C. Xu, “Hierarchical cross-modal
talking face generation with dynamic pixel-wise loss,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recogni-
tion, 2019, pp. 7832–7841.
[20] H. Zhou, Y. Liu, Z. Liu, P. Luo, and X. Wang, “Talking face generation
by adversarially disentangled audio-visual representation,” in Proceed-
ings of the AAAI conference on artificial intelligence, vol. 33, no. 01,
2019, pp. 9299–9306.
[21] L. Yu, J. Yu, M. Li, and Q. Ling, “Multimodal inputs driven talking
face generation with spatial–temporal dependency,” IEEE Transactions
on Circuits and Systems for Video Technology, vol. 31, no. 1, pp. 203–
216, 2020.
[22] D. Das, S. Biswas, S. Sinha, and B. Bhowmick, “Speech-driven facial
animation using cascaded gans for learning of motion and texture,” in
Computer Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part XXX 16.
Springer, 2020,
pp. 408–424.
[23] Y. Sun, H. Zhou, Z. Liu, and H. Koike, “Speech2talking-face: Inferring
and driving a face with synchronized audio-visual representation.” in
IJCAI, vol. 2, 2021, p. 4.
[24] F. Yin, Y. Zhang, X. Cun, M. Cao, Y. Fan, X. Wang, Q. Bai,
B. Wu, J. Wang, and Y. Yang, “Styleheat: One-shot high-resolution
editable talking face generation via pre-trained stylegan,” in European
conference on computer vision.
Springer, 2022, pp. 85–101.
[25] Y. Zhou, Z. Xu, C. Landreth, E. Kalogerakis, S. Maji, and K. Singh,
“Visemenet: Audio-driven animator-centric speech animation,” ACM
Transactions on Graphics (TOG), vol. 37, no. 4, pp. 1–10, 2018.
[26] E. Zakharov, A. Shysheya, E. Burkov, and V. Lempitsky, “Few-
shot adversarial learning of realistic neural talking head models,” in
Proceedings of the IEEE/CVF international conference on computer
vision, 2019, pp. 9459–9468.
[27] Y. Lu, J. Chai, and X. Cao, “Live speech portraits: real-time photore-
alistic talking-head animation,” ACM Transactions on Graphics (ToG),
vol. 40, no. 6, pp. 1–17, 2021.
[28] S. Taylor, T. Kim, Y. Yue, M. Mahler, J. Krahe, A. G. Rodriguez,
J. Hodgins, and I. Matthews, “A deep learning approach for generalized
speech animation,” ACM Transactions On Graphics (TOG), vol. 36,
no. 4, pp. 1–11, 2017.
[29] S. Suwajanakorn, S. M. Seitz, and I. Kemelmacher-Shlizerman, “Syn-
thesizing obama: learning lip sync from audio,” ACM Transactions on
Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
[30] J. Thies, M. Elgharib, A. Tewari, C. Theobalt, and M. Nießner, “Neural
voice puppetry: Audio-driven facial reenactment,” in Computer Vision–
ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28,
2020, Proceedings, Part XVI 16.
Springer, 2020, pp. 716–731.

<!-- page 11 -->
11
[31] R. Yi, Z. Ye, J. Zhang, H. Bao, and Y.-J. Liu, “Audio-driven talking face
video generation with learning-based personalized head pose,” arXiv
preprint arXiv:2002.10137, 2020.
[32] Y. Ren, G. Li, Y. Chen, T. H. Li, and S. Liu, “Pirenderer: Controllable
portrait image generation via semantic neural rendering,” in Proceed-
ings of the IEEE/CVF international conference on computer vision,
2021, pp. 13 759–13 768.
[33] W. Zhang, X. Cun, X. Wang, Y. Zhang, X. Shen, Y. Guo, Y. Shan,
and F. Wang, “Sadtalker: Learning realistic 3d motion coefficients
for stylized audio-driven single image talking face animation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 8652–8661.
[34] T. Karras, T. Aila, S. Laine, A. Herva, and J. Lehtinen, “Audio-driven
facial animation by joint end-to-end learning of pose and emotion,”
ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1–12, 2017.
[35] H. X. Pham, S. Cheung, and V. Pavlovic, “Speech-driven 3d facial ani-
mation with implicit emotional awareness: A deep learning approach,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition workshops, 2017, pp. 80–88.
[36] H. X. Pham, Y. Wang, and V. Pavlovic, “End-to-end learning for
3d facial animation from speech,” in Proceedings of the 20th ACM
International Conference on Multimodal Interaction, 2018, pp. 361–
365.
[37] D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black, “Cap-
ture, learning, and synthesis of 3d speaking styles,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2019, pp. 10 101–10 111.
[38] P. Tzirakis, A. Papaioannou, A. Lattas, M. Tarasiou, B. Schuller,
and S. Zafeiriou, “Synthesising 3d facial motion from “in-the-wild”
speech,” in 2020 15th IEEE International Conference on Automatic
Face and Gesture Recognition (FG 2020).
IEEE, 2020, pp. 265–272.
[39] A. Richard, M. Zollh¨ofer, Y. Wen, F. De la Torre, and Y. Sheikh,
“Meshtalk: 3d face animation from speech using cross-modality disen-
tanglement,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2021, pp. 1173–1182.
[40] Z. Peng, Y. Luo, Y. Shi, H. Xu, X. Zhu, H. Liu, J. He, and Z. Fan,
“Selftalk: A self-supervised commutative training diagram to compre-
hend 3d talking faces,” in Proceedings of the 31st ACM International
Conference on Multimedia, 2023, pp. 5292–5301.
[41] K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao,
and C. C. Loy, “Mead: A large-scale audio-visual dataset for emotional
talking-face generation,” in European Conference on Computer Vision.
Springer, 2020, pp. 700–717.
[42] S. R. Livingstone and F. A. Russo, “The ryerson audio-visual database
of emotional speech and song (ravdess): A dynamic, multimodal set
of facial and vocal expressions in north american english,” PloS one,
vol. 13, no. 5, p. e0196391, 2018.
[43] S. E. Eskimez, Y. Zhang, and Z. Duan, “Speech driven talking face
generation from a single image and an emotion condition,” IEEE
Transactions on Multimedia, vol. 24, pp. 3480–3490, 2021.
[44] B. Liang, Y. Pan, Z. Guo, H. Zhou, Z. Hong, X. Han, J. Han,
J. Liu, E. Ding, and J. Wang, “Expressive talking head generation
with granular audio-visual control,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
3387–3396.
[45] S. Sinha, S. Biswas, R. Yadav, and B. Bhowmick, “Emotion-
controllable generalized talking face generation,” arXiv preprint
arXiv:2205.01155, 2022.
[46] Y. Ma, S. Wang, Z. Hu, C. Fan, T. Lv, Y. Ding, Z. Deng, and
X. Yu, “Styletalk: One-shot talking head generation with controllable
speaking styles,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 37, no. 2, 2023, pp. 1896–1904.
[47] Z. Sun, Y.-H. Wen, T. Lv, Y. Sun, Z. Zhang, Y. Wang, and Y.-J. Liu,
“Continuously controllable facial expression editing in talking face
videos,” IEEE Transactions on Affective Computing, 2023.
[48] Z. Peng, H. Wu, Z. Song, H. Xu, X. Zhu, J. He, H. Liu, and
Z. Fan, “Emotalk: Speech-driven emotional disentanglement for 3d face
animation,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 20 687–20 697.
[49] Q. He, X. Ji, Y. Gong, Y. Lu, Z. Diao, L. Huang, Y. Yao, S. Zhu,
Z. Ma, S. Xu et al., “Emotalk3d: high-fidelity free-view synthesis
of emotional 3d talking head,” in European Conference on Computer
Vision.
Springer, 2024, pp. 55–72.
[50] S. Xu, G. Chen, Y.-X. Guo, J. Yang, C. Li, Z. Zang, Y. Zhang, X. Tong,
and B. Guo, “Vasa-1: Lifelike audio-driven talking faces generated in
real time,” arXiv preprint arXiv:2404.10667, 2024.
[51] S. Tan, B. Ji, Y. Ding, and Y. Pan, “Say anything with any style,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38,
no. 5, 2024, pp. 5088–5096.
[52] S. Tan, B. Ji, M. Bi, and Y. Pan, “Edtalk: Efficient disentanglement for
emotional talking head synthesis,” arXiv preprint arXiv:2404.01647,
2024.
[53] C. Ma, S. Tan, J. Wei, and Y. Pan, “Goes: 3d gaussian-based one-
shot head animation with any emotion and any style,” in Proceedings
of the 33rd ACM International Conference on Multimedia, 2025, pp.
9578–9587.
[54] S. Tan and B. Ji, “Edtalk++: Full disentanglement for controllable
talking head synthesis,” arXiv preprint arXiv:2508.13442, 2025.
[55] S. Tan, B. Gong, Z. Liu, Y. Wang, X. Chen, Y. Feng, and H. Zhao,
“Animate-x++: Universal character image animation with dynamic
backgrounds,” arXiv preprint arXiv:2508.09454, 2025.
[56] B. Ji, Y. Pan, Z. Liu, S. Tan, and X. Yang, “Sport: From zero-
shot prompts to real-time motion generation,” IEEE Transactions on
Visualization and Computer Graphics, 2025.
[57] S. Tan, B. Ji, and Y. Pan, “Flowvqtalker: High-quality emotional
talking face generation through normalizing flow and quantization,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 26 317–26 327.
[58] P. Witzig, B. Solenthaler, M. Gross, and R. Wampfler, “Emospacetime:
Decoupling emotion and content through contrastive learning for
expressive 3d speech animation,” in Proceedings of the 17th ACM
SIGGRAPH Conference on Motion, Interaction, and Games, 2024, pp.
1–12.
[59] S. Tan, B. Gong, Y. Feng, K. Zheng, D. Zheng, S. Shi, Y. Shen, J. Chen,
and M. Yang, “Mimir: Improving video diffusion models for precise
text understanding,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2025.
[60] S. Tan, B. Gong, Y. Wei, S. Zhang, Z. Liu, D. Zheng, J. Chen, Y. Wang,
H. Ouyang, K. Zheng, and Y. Shen, “Synmotion: Semantic-visual
adaptation for motion customized video generation,” arXiv preprint
arXiv:2506.23690, 2025.
[61] S. Tan, B. Gong, X. Wang, S. Zhang, D. Zheng, R. Zheng, K. Zheng,
J. Chen, and M. Yang, “Animate-x: Universal character image anima-
tion with enhanced motion representation,” in International Conference
on Learning Representations, 2025.
[62] S. Tan, B. Gong, B. Ji, and Y. Pan, “Fixtalk: Taming identity leakage for
high-quality talking head generation in extreme cases,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2025.
[63] Y. Pan, R. Zhang, S. Cheng, S. Tan, Y. Ding, K. Mitchell, and X. Yang,
“Emotional voice puppetry,” IEEE Transactions on Visualization and
Computer Graphics, vol. 29, no. 5, pp. 2527–2535, 2023.
[64] Y. Pan, S. Tan, S. Cheng, Q. Lin, Z. Zeng, and K. Mitchell, “Expressive
talking avatars,” IEEE Transactions on Visualization and Computer
Graphics, vol. 30, no. 5, pp. 2538–2548, 2024.
[65] Y. Pan, C. Liu, S. Xu, S. Tan, and J. Yang, “Vasa-rig: Audio-driven
3d facial animation with ‘live’mood dynamics in virtual reality,” IEEE
Transactions on Visualization and Computer Graphics, 2025.
[66] X. Liu, Y. Xu, Q. Wu, H. Zhou, W. Wu, and B. Zhou, “Semantic-aware
implicit neural audio-driven video portrait generation,” in European
conference on computer vision.
Springer, 2022, pp. 106–125.
[67] S. Shen, W. Li, Z. Zhu, Y. Duan, J. Zhou, and J. Lu, “Learning dynamic
facial radiance fields for few-shot talking head synthesis,” in European
conference on computer vision.
Springer, 2022, pp. 666–682.
[68] S. Yao, R. Zhong, Y. Yan, G. Zhai, and X. Yang, “Dfa-nerf: Person-
alized talking head generation via disentangled face attributes neural
rendering,” arXiv preprint arXiv:2201.00791, 2022.
[69] J. Tang, K. Wang, H. Zhou, X. Chen, D. He, T. Hu, J. Liu, G. Zeng,
and J. Wang, “Real-time neural radiance talking portrait synthesis via
audio-spatial decomposition,” arXiv preprint arXiv:2211.12368, 2022.
[70] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[71] J. Li, J. Zhang, X. Bai, J. Zhou, and L. Gu, “Efficient region-aware
neural radiance fields for high-fidelity talking portrait synthesis,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 7568–7578.
[72] W. Yu, Y. Fan, Y. Zhang, X. Wang, F. Yin, Y. Bai, Y.-P. Cao,
Y. Shan, Y. Wu, Z. Sun et al., “Nofa: Nerf-based one-shot facial avatar
reconstruction,” in ACM SIGGRAPH 2023 Conference Proceedings,
2023, pp. 1–12.
[73] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain,
and M. Nießner, “Gaussianavatars: Photorealistic head avatars with

<!-- page 12 -->
12
rigged 3d gaussians,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 20 299–20 309.
[74] Z. Zhou, F. Ma, H. Fan, and Y. Yang, “Headstudio: Text to an-
imatable head avatars with 3d gaussian splatting,” arXiv preprint
arXiv:2402.06149, 2024.
[75] H. Yu, Z. Qu, Q. Yu, J. Chen, Z. Jiang, Z. Chen, S. Zhang, J. Xu, F. Wu,
C. Lv et al., “Gaussiantalker: Speaker-specific talking head synthesis
via 3d gaussian splatting,” arXiv preprint arXiv:2404.14037, 2024.
[76] J. Wang, J.-C. Xie, X. Li, F. Xu, C.-M. Pun, and H. Gao, “Gaussian-
head: High-fidelity head avatars with learnable gaussian derivation,”
arXiv preprint arXiv:2312.01632, 2023.
[77] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
“Talkinggaussian: Structure-persistent 3d talking head synthesis via
gaussian splatting,” arXiv preprint arXiv:2404.15264, 2024.
[78] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero, “Learning a
model of facial shape and expression from 4d scans.” ACM Trans.
Graph., vol. 36, no. 6, pp. 194–1, 2017.
[79] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. D.
Mello, O. Gallo, L. Guibas, J. Tremblay, S. Khamis, T. Karras, and
G. Wetzstein, “Efficient geometry-aware 3D generative adversarial
networks,” in CVPR, 2022.
[80] S. Yang, L. Jiang, Z. Liu, and C. C. Loy, “Pastiche master: Exemplar-
based high-resolution portrait style transfer,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 7693–7702.
[81] E. Richardson, Y. Alaluf, O. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro,
and D. Cohen-Or, “Encoding in style: a stylegan encoder for image-
to-image translation,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2021, pp. 2287–2296.
[82] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture
for generative adversarial networks,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2019, pp.
4401–4410.
[83] S. Yang, L. Jiang, Z. Liu, and C. C. Loy, “Vtoonify: Controllable high-
resolution portrait video style transfer,” ACM Transactions on Graphics
(TOG), vol. 41, no. 6, pp. 1–15, 2022.
[84] S. Tan, B. Ji, and Y. Pan, “Style2talker: High-resolution talking head
generation with emotion style and art style,” in Proceedings of the
AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp.
5079–5087.
[85] Y. Men, H. Liu, Y. Yao, M. Cui, X. Xie, and Z. Lian, “3dtoonify:
Creating your high-fidelity 3d stylized avatar easily from 2d portrait
images,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 10 127–10 137.
[86] R. Shao, J. Sun, C. Peng, Z. Zheng, B. Zhou, H. Zhang, and Y. Liu,
“Control4d: Efficient 4d portrait editing with text,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2024, pp. 4556–4567.
[87] Z. Chen, Y. Yan, S. Liu, Y. Cheng, W. Zhao, L. Li, M. Bi, and
X. Yang, “Revealing directions for text-guided 3d face editing,” IEEE
Transactions on Multimedia, 2025.
[88] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Surface splatting,”
in Proceedings of the 28th annual conference on Computer graphics
and interactive techniques, 2001, pp. 371–378.
[89] A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen,
R. Prenger, S. Satheesh, S. Sengupta, A. Coates et al., “Deep
speech: Scaling up end-to-end speech recognition,” arXiv preprint
arXiv:1412.5567, 2014.
[90] Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, “Accurate
3d face reconstruction with weakly-supervised learning: From single
image to image set,” in IEEE Computer Vision and Pattern Recognition
Workshops, 2019.
[91] T. Baltruˇsaitis, P. Robinson, and L.-P. Morency, “Openface: an open
source facial behavior analysis toolkit,” in 2016 IEEE winter conference
on applications of computer vision (WACV).
IEEE, 2016, pp. 1–10.
[92] Z. Peng, W. Hu, Y. Shi, X. Zhu, X. Zhang, H. Zhao, J. He, H. Liu, and
Z. Fan, “Synctalk: The devil is in the synchronization for talking head
synthesis,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 666–676.
[93] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time
style transfer and super-resolution,” in Computer Vision–ECCV 2016:
14th European Conference, Amsterdam, The Netherlands, October 11-
14, 2016, Proceedings, Part II 14.
Springer, 2016, pp. 694–711.
[94] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[95] C.-H. Lee, Z. Liu, L. Wu, and P. Luo, “Maskgan: Towards diverse and
interactive facial image manipulation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp.
5549–5558.
[96] J. N. Pinkney and D. Adler, “Resolution dependent gan interpolation
for controllable image synthesis between domains,” arXiv preprint
arXiv:2010.05334, 2020.
[97] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter,
“Gans trained by a two time-scale update rule converge to a local
nash equilibrium,” Advances in neural information processing systems,
vol. 30, 2017.
[98] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
[99] J. S. Chung and A. Zisserman, “Out of time: automated lip sync
in the wild,” in Computer Vision–ACCV 2016 Workshops: ACCV
2016 International Workshops, Taipei, Taiwan, November 20-24, 2016,
Revised Selected Papers, Part II 13.
Springer, 2017, pp. 251–263.
[100] S. Cheng, C. Ma, and Y. Pan, “Stylizedfacepoint: Facial landmark
detection for stylized characters,” in Proceedings of the 32nd ACM
International Conference on Multimedia, 2024, pp. 8072–8080.
[101] D. Meng, X. Peng, K. Wang, and Y. Qiao, “Frame attention networks
for facial expression recognition in videos,” in 2019 IEEE international
conference on image processing (ICIP).
IEEE, 2019, pp. 3866–3870.
[102] Z. Zhang, L. Li, Y. Ding, and C. Fan, “Flow-guided one-shot talking
face generation with a high-resolution audio-visual dataset,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021, pp. 3661–3670.
[103] B. Azari and A. Lim, “Emostyle: One-shot facial expression editing us-
ing continuous emotion parameters,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2024, pp.
6385–6394.
VIII. BIOGRAPHY SECTION
Chuhang Ma is a member of Character Lab,
Shanghai Jiao Tong University, Shanghai, China. He
received the B.E. degree in Artificial Intelligence
from Shanghai Jiao Tong University. He is currently
pursuing the Ph.D. degree in Computer Science and
Engineering at Shanghai Jiao Tong University. His
research interest includes computer vision and 3D
facial animation.
Shuai Tan is a member of Character Lab, Shang-
hai Jiao Tong University, Shanghai, China. He re-
ceived the B.S. degree in software engineering from
Sichuan University. He is currently pursuing the
Ph.D. degree in Computer Science and Engineer-
ing at Shanghai Jiao Tong University. His research
interest includes computer vision and multi-modal
learning.

<!-- page 13 -->
13
Ye Pan is currently an Associate Professor with
Shanghai Jiao Tong University. Her research in-
terests include AR/VR, avatars/characters, 3D an-
imations, HCI, and computer graphics. Previously,
she was an Associate Research Scientist in AR/VR
at Disney Research Los Angeles. She received the
B.Sc. degree in communication and information
engineering from Purdue/UESTC in 2010 and the
Ph.D. degree in computer graphics from the Univer-
sity College London (UCL) in 2015. She has served
as Associate Editor of the International Journal of
Human Computer Studies, and a regular member of IEEE virtual reality
program committees.
Jiaolong Yang is currently a senior researcher at
Microsoft Research Asia, Beijing, China. He re-
ceived the dual Ph.D. degrees in Computer Sci-
ence and Engineering from the Australian National
University and Beijing Institute of Tech nology
in 2016. His research interests include 3D vision
for human face and body. He serve as the pro-
gram committee member/reviewer for major com-
puter vision conferences and journals including
CVPR/ICCV/ECCV/TPAMI/IJCV, the Area Chair
for CVPR/ICCV/ECCV/WACV/MM, and the Asso-
ciate Editor for the International Journal on Computer Vision (IJCV).
Xin Tong received the BS and master’s degrees in
computer science from Zhejiang University in 1993
and 1996, respectively, and the PhD degree in com-
puter graphics from Tsinghua University in 1999.
He is currently a principal researcher with Internet
Graphics Group, Microsoft Research Asia. His PhD
thesis is about hardware assisted volume rendering.
His research interests include appearance modeling
and rendering, texture synthesis, and image based
modeling and rendering.
