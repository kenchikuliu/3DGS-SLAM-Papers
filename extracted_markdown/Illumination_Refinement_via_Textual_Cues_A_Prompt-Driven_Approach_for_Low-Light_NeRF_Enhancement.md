<!-- page 1 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
2019
Illumination Reﬁnement via Textual Cues:
A Prompt-Driven Approach for Low-Light NeRF
Enhancement
Xinrui Ju* , Yang Zou* , Xingyuan Li , Zirui Wang , Jun Ma , Zhiying Jiang ,
and Jinyuan Liu , Member, IEEE
Abstract—In the realm of 3D scene modeling and rendering,
the emergence of Neural Radiance Fields (NeRF) represents a sig-
niﬁcant leap forward. However, NeRF’s rendering performance
suﬀers signiﬁcantly when rendering images under low-light con-
ditions. Existing approaches are optimized by enhancing low-light
input images and combining NeRF models, but still fail to address
the issues of multiview consistency and image quality. To address
these challenges, our research introduces a textual constraint-
prompted enhancement method that facilitates low-light image
brightening and new view synthesis in an unsupervised man-
ner. Speciﬁcally, we devise a semantic calibration strategy that
employs positive and negative prompts to motivate and penalize
the network towards attributes associated with high-quality
images and exploits the capability of visual language models
in semantic parsing to align the generated images with textual
descriptors to improve image generation quality. In addition, to
address the multiview consistency problem, we propose a two-
layer optimization strategy, where the semantic cue optimization
in the upper layer and the new view generation in the lower layer
interact with each other to achieve a balance between luminance
consistency and structural integrity by combining these improved
images with text-driven semantic features. Comprehensive tests
on two datasets with diﬀerent resolutions, LOM and LLFF, show
that our approach outperforms existing methods by signiﬁcantly
improving the brightness and clarity of low-light images to state-
of-the-art while preserving the natural appearance and details.
Received 28 April 2025; revised 5 July 2025 and 8 August 2025;
accepted 24 August 2025. Date of publication 29 August 2025; date of current
version 5 February 2026. This work was supported in part by the National
Natural Science Foundation of China under Grant 62302078 and Grant
62372080 and in part by China Postdoctoral Science Foundation under Grant
2023M730741. This article was recommended by Associate Editor C. Chen.
(Xinrui Ju and Yang Zou contributed equally to this work.) (Corresponding
author: Jinyuan Liu.)
Xinrui Ju is with the DUT-RU International School of Information Science
and Engineering, Dalian University of Technology, Dalian 116024, China,
and also with the Department of Computer Science, City University of Hong
Kong, Hong Kong (e-mail: juxinrui1021@163.com).
Yang Zou is with the School of Computer Science, Northwestern Polytech-
nical University, Xi’an 710129, China (e-mail: archerv2@mail.nwpu.edu.cn).
Xingyuan Li is with the DUT-RU International School of Information
Science and Engineering, Dalian University of Technology, Dalian 116024,
China, and also with the College of Computer Science and Technology,
Zhejiang University, Hangzhou 310058, China (e-mail: xingyuan lxy@
163.com).
Zirui
Wang
and
Jun
Ma
are
with
the
DUT-RU
International
School of Information Science and Engineering, Dalian University of
Technology, Dalian 116024, China (e-mail: ziruiwang0625@gmail.com;
junma.work812@gmail.com).
Zhiying Jiang is with the College of Information Science and Tech-
nology,
Dalian
Maritime
University,
Dalian
116024,
China
(e-mail:
zyjiang0630@mail.dlut.edu.cn).
Jinyuan Liu is with the School of Software Technology, Dalian University
of Technology, Dalian 116024, China (e-mail: atlantis918@hotmail.com).
Digital Object Identiﬁer 10.1109/TCSVT.2025.3604241
Index Terms—Low-light enhancement, 3D vision.
I. INTRODUCTION
N
EURAL Radiance Fields (NeRF) [1] represent a pivotal
advancement in the understanding of 3D scenes from
2D images. This is achieved by learning scene representations
via implicit functions, which are parameterized through the
utilization of multi-layer perceptrons.These perceptions are
optimized through the evaluation of colorimetric discrepan-
cies in the input views, necessitating the use of high-quality
images as a foundational requirement for NeRF’s optimal
performance. Essentially, the eﬃcacy of NeRF models [2], [3]
is contingent upon the input images’ ability to accurately and
clearly depict the illumination and coloration of the scenes.
Nevertheless, this requirement presents a signiﬁcant challenge,
akin to obstacles faced in other domains such as image
processing [4], [5], [6], [7], [8], object detection [9], [10],
[11], etc. The acquisition of images in real-world scenarios is
frequently subject to uncontrollable variables, with low-light
conditions being particularly detrimental. These conditions
severely impair the performance of these technologies, thereby
limiting their applicability in scenarios where lighting cannot
be adequately controlled. The inherent limitations of NeRF
come from its viewer-centric methodology, which calculates
the emission of light from a single perspective and generates
images by simulating the process of light propagating from
a certain point in the scene to the observer. This approach
simpliﬁes the propagation path of light to a certain extent,
treating it only as a reﬂection process aﬀected by ambient
light, while taking into account some refraction and absorption
factors, but ignores the complex interactions between light
and various components in the scene [12], such as sur-
face scattering, local reﬂection, and material-related lighting
changes. Therefore, these ignored interactions between light
and materials become particularly critical under poor lighting
conditions. NeRF cannot eﬀectively capture and restore these
complex lighting details during the modeling process, resulting
in deviations in the reconstruction eﬀect when faced with
scenes with insuﬃcient lighting or uneven light sources in the
real world.
An intuitive approach to addressing this issue is to enhance
the low-light input images prior [13], [14], [15] to training
the NeRF model. While this strategy can improve image
brightness to some extent, existing low-light enhancement
1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
2020
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
Fig. 1.
Schematic of the proposed prompt-driven enhancement method for
low-light neural radiance scenarios. Our approach uses textual cues to guide
the enhancement and subsequently reﬁnes the NeRF network to generate high-
quality, well-lit new views. The ﬁgure demonstrates the eﬀectiveness of our
approach through a visual comparison depicting the role of textual constraints
in guiding the semantic calibration process.
models still suﬀer from a lack of multi-view consistency. They
often fail to ensure uniformity in brightness variations across
diﬀerent viewpoints. Pixel-to-pixel 2D image enhancement
methods do not guarantee accurate NeRF estimation, and
independently applying enhancement across multiple views
can disrupt the consistency of 3D geometry.
Recently, AME-NeRF [16] was proposed to address low-
light reconstruction by using a bi-level optimization strategy
to jointly optimize exposure correction and novel view syn-
thesis. However, its exposure evaluation relies on manually
set coeﬃcients rather than learned ones. Aleth-NeRF [17]
introduces “Concealing Fields” to handle low-light and over-
exposed scenes, generating normal-light views, but struggles
under complex, uneven lighting or shadows. LITA-GS [18]
enhances clarity in low-light with a progressive denoising
module and supports fast training, yet incorrect structural
details may misguide the optimization process and reduce
overall consistency.
To address the problem of degraded rendering performance
of NeRF under low-light conditions, we propose an enhance-
ment method based on textual constraint cues. The method
makes full use of the visual language model’s capability in
semantic parsing, so that the generated images can maintain
the consistency between the brightness enhancement and the
semantic descriptions of high-quality images. As shown in
Figure 1, our approach improves the perceived quality of the
generated image by designing positive and negative prompts
that guide the network to enhance key semantic attributes (e.g.,
well-lit, clear and sharp) in the image while suppressing low-
quality features (e.g., dim, blur). To achieve ﬁner control,
we introduce a sigmoid-like dynamic probabilistic model
that gradually increases the inﬂuence of prompts during the
training process to achieve goal-directed semantic reﬁnement,
which improves the stability of image enhancement and the
quality of image generation.
In addition, we design a Bi-Level Optimization Framework
to address the challenge of coordinating image enhancement
and 3D reconstruction tasks. The upper-level task focuses on
a semantically guided image enhancement network, which is
optimized through text-based prompts, while the lower-level
task consists of a NeRF-based 3D reconstruction module that
synthesizes novel views from the enhanced images. A proxy
supervision mechanism is introduced to establish coupling
between the two tasks, where the enhanced results from the
upper level inﬂuence the performance of NeRF reconstruc-
tion, and the quality of the view generated by the NeRF
in turn guides the optimization of the enhancement network.
Through this closed-loop training strategy, we achieve a bal-
ance between brightness consistency and structural integrity,
signiﬁcantly improving 3D modeling and image rendering
under low-light conditions. In summary, our contributions can
be summarized as follows:
• We propose advanced brightness enhancement techniques
with textual information as prompts, which can eﬀectively
improve the modeling and rendering of images in low-
light conditions. By leveraging semantic cues, our method
enhances the visibility of images while preserving impor-
tant scene details.
• We design a bi-level optimization framework for 3D
reconstruction. The framework integrates a text-guided
image enhancement network with an improved NeRF
architecture, eﬀectively balancing the brightness consis-
tency between diﬀerent views and the geometric and
structural integrity during reconstruction.
• Extensive results on LOM [17] and LLFF [19] demon-
strate that our method not only improves visual quality
under low-light conditions, but also enhances the accuracy
and robustness of 3D reconstruction tasks.
II. RELATED WORK
A. Neural Radiance Field in Weak Conditions
NeRF [1] has attracted wide attention for its realistic ren-
dering eﬀects and powerful scene representation capabilities.
Deep learning technology is used to extract the geometry
and texture information of objects from images in multiple
perspectives, and a scene function is optimized by adopting
a volume rendering strategy on the corresponding light to
achieve the generation of new perspective images. However,
the excellent performance of NeRF is very dependent on a
series of degradation factors such as the number of input per-
spectives, the accuracy of camera poses, the lighting conditions
of the input images, illumination variations, and blur. Actually,
in actual image acquisition, it is diﬃcult to meet the above
conditions, which will inﬂuence the eﬀect of NeRF.
To address limited viewpoints and inaccurate camera poses,
PixelNeRF [20] predicts neural scene representations from
sparse images using convolutional conditioning and learned
priors. RegNeRF [21] introduces geometry and appearance
regularization for unseen views and anneals light sampling
to handle sparse scenes. DietNeRF [22] adds semantic consis-
tency loss for more realistic rendering under sparse data. Many
methods have also shown excellent results for input images
under low-light conditions and image degradation problems
[16], [17], [18]. Deblur-NeRF [23] introduces a Deformable
Sparse Kernel (DSK) module to explicitly simulate the physi-
cal blur process. By synthesizing a blurred image to match
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2021
the input image, the NeRF and DSK modules are jointly
optimized to achieve the goal of restoring a clear NeRF from
a blurred input. LLNeRF [24] enhances scene representation
in an unsupervised manner and synthesizes new views of
normal illumination directly from sRGB low-light images.
RawNeRF [25] is trained speciﬁcally on unprocessed RAW
images, improving the quality and authenticity of synthetic
view images from the data source level.
B. Low-Light Image Enhancement
In the real world, there are many factors that aﬀect image
acquisition, and if the desired image cannot be acquired,
image processing [13], [14], [26], [27], [28], [29], [30],
[31]is required. Low-light conditions [32] present signiﬁcant
challenges for visual systems as an intrinsic aspect of our
natural environment. Low-Light Image Enhancement (LLIE)
techniques are developed to address these challenges, aiming
to amplify image brightness and detail where lighting is sub-
optimal. Traditional enhancement methods, such as gray level
transformation [33], histogram equalization [34], and Retinex-
based algorithms [35], have historically mitigated low-light
issues by manipulating the grayscale values to improve per-
ceptual visibility. However, traditional methods often face
limitations like over-enhancement, noise ampliﬁcation, and
poor adaptation to varying low-light complexities across diﬀer-
ent scenes. Operating in a heuristic-driven manner, they lack
adaptive learning to incorporate contextual information, lim-
iting their robustness and eﬀectiveness, especially in detailed
or complex scenes.
With the rapid development of deep learning, contemporary
methodologies enhance images by extracting and learning
from features observed in normally lit conditions. Notable
among these are LLNet [36], a deep autoencoder-based net-
work, SKF [37], which utilizes sparse coding frameworks, and
NeRCo [38], an approach that implicitly conditions the image
enhancement process on contextual data. These advanced
methods beneﬁt from extensive training on large datasets,
enabling them to achieve enhancements that are both signiﬁ-
cant and perceptually pleasing.
C. Prompt-Based Learning
The intersection of natural language processing and com-
puter vision has burgeoned with the advent of prompt-based
learning paradigms. One seminal work in this domain is CLIP
[39], which leverages contrastive learning to bridge textual and
visual representations, facilitating a range of tasks from image
classiﬁcation to complex similarity assessments through its
text-image processing capabilities. The versatility of CLIP has
been demonstrated across a spectrum of high-level applications
[40], [41], [42].
Building upon this, CoOp [43] reﬁnes the approach by
optimizing a task-speciﬁc objective function that enables auto-
matic learning of relevant prompts, enhancing task-speciﬁc
performance. Further extending this framework, CoCoOp
[44] introduces Meta-Net, a lightweight network designed to
improve generalization by conditionally adapting prompts. In
the realm of image enhancement, CLIP-LIT [45] employs a
frozen CLIP model to compute and utilize text-image sim-
ilarity for the iterative training of an enhancement network.
Dynamic Prompt Learning (DPL) [46] is proposed for image
editing by forcing cross-attentional maps to focus on the
correct noun vocabulary in textual cues. Similarly, PromptIR
[47] presents a pioneering approach to blind image restoration.
This system utilizes degradation-speciﬁc prompts to deftly
guide restoration networks toward targeted outcomes, exempli-
fying the potential of prompt-based methods in adaptive image
recovery.
III. METHOD
In this section, we present the proposed method where
we use prompt-based image pre-adjustment and adaptive
semantic calibration to generate preprocessed images and
semantic representations for improving image quality and 3D
reconstruction in low-light conditions. The method utilises
the CLIP-LIT framework for semantic understanding and
NeRF-based volume rendering as shown in Figure 2. A bi-
level optimisation strategy is then introduced where prompt
learning and enhancement networks are used alternatively in
the ﬁrst stage to achieve semantic accuracy and visual integrity
of low-light images. The bi-level optimisation framework
combines image enhancement with 3D reconstruction, where
the enhanced image informs the NeRF network to generate
new views. By alternately optimising the enhancement and
reconstruction networks, the method ensures high-quality 3D
reconstruction while maintaining semantic and structural con-
sistency.
A. Prompt-Based Image Pre-Tuning
Our framework employs the CLIP-LIT [45] module for
the pre-tuning of images in low-light conditions. CLIP-LIT
is a two-stage approach that leverages the semantic insight
provided by the CLIP model to inform the enhancement
of underexposed images. This approach alternates between
updating the prompt learning framework and the enhancement
network until visually pleasing results are achieved. Our usage
of the text encoder during training diﬀers from that of the
original CLIP-LIT. CLIP-LIT ﬁrst unlocks the text encoder
to compute the initial loss, and then freezes it for the CLIP
loss. In contrast, our method keeps the text encoder unlocked
throughout the entire training process to support a dual-level
optimization strategy.
In the initial phase, CLIP-LIT generates prompt pairs to
diﬀerentiate between backlit and well-lit conditions and pro-
ceeds to train an enhancement network. This training begins
by initializing positive Tp and negative Tn prompt vectors in
RN×512. Then encode the images using CLIP’s image encoder
Φimage to derive their latent representations.
The learning process leverages the binary cross-entropy loss
for initial prompt classiﬁcation and the similarity in CLIP’s
latent space:
Linitial = −(y log(ˆy) + (1 −y) log(1 −ˆy)),
(1)
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
2022
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
Fig. 2. Overall architecture of the proposed enhancement methodology. The workﬂow initiates with the preprocessing of the low-light input images, which
are then fed into an augmented NeRF system for volumetric rendering. Subsequently, the rendering is reﬁned under the auspices of a bi-level optimization
model, culminating in the generation of the desired outcome with improved illumination and visual ﬁdelity.
where y is the label of the image type and Φtext is the text
encoder of the CLIP model, while the ˆy is the predicted label
which is deﬁned as:
ˆy =
ecos(Φimage(I),Φtext(Tp))
P
i∈{p,n} ecos(Φimage(I),Φtext(Ti)) .
(2)
The network is reﬁned with a CLIP-aware loss, combining
identity loss Lidentity and CLIP-Enhance loss Lclip:
Lidentity =
4
X
l=0
αl · ||Φl
image(Ib) −Φl
image(Ilt)||2,
(3)
Lclip =
ecos(Φimage(It),Φtext(Tn))
P
i∈{n,p} ecos(Φimage(It),Φtext(Ti)) ,
(4)
where αl are layer weights in the CLIP model, Ib is the backlit
image, and Ilt the enhanced image. This ensures both semantic
ﬁdelity and visual integrity in the enhanced output.
In the second stage, the network is iteratively optimized
using margin ranking loss, with a focus on improving image
illumination rather than overall content features. The loss
ensures an increased dissimilarity in similarity scores between
well-lit and backlit images in the CLIP embedding space,
while also constraining the similarity scores of enhanced
images with those of well-lit images. CLIP-LIT incorpo-
rates previous enhancement results into the loss calculation
to progressively reﬁne prompts, enhancing the recognition
and enhancement of subtle luminance variations under low-
light conditions. Therefore, the CLIP-LIT module acts as a
“semantic evaluator” during training, guiding the optimization
of the image enhancement network.
B. Adaptive Semantic Calibration
In order to optimize the quality of images generated by
the neural radiance ﬁeld model, we propose adaptive semantic
calibration, which combines prompt-based loss and stochas-
tic structural similarity (S3IM) loss, modeled by a Sigmoid
function, to ensure that the image performs well in terms of
semantic accuracy and structural ﬁdelity.
1) Prompt-Based Loss: The prompt-based loss utilizes the
CLIP model’s proﬁciency in semantic parsing to align the
generated images with textual descriptors. The design of the
prompts is based on an intuitive understanding of image
quality semantics, combined with CLIP’s prior capability
in aligning visual and linguistic concepts. Positive prompts
such as “A well-lit photo” and “A clear and sharp photo”
represent typical semantic attributes of high-quality images
and encourage the network to optimize toward desirable visual
properties. In contrast, negatively connoted prompts like “A
dim photo” and “A blur photo” describe common issues in
low-quality images and serve as deterrents by penalizing the
presence of such undesirable features in the NeRF outputs.
To quantify this alignment, we deﬁne the prompt loss in
probabilistic terms as follows:
Lprompt =
exp(Σpos)
exp(Σpos) + exp(Σneg),
(5)
where
Σpos =
X
p∈P
cos(Φimage(It), Φtext(Tp)),
Σneg =
X
n∈N
cos(Φimage(It), Φtext(Tn)),
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 5 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2023
with P and N denoting the sets of positive and negative
prompts, respectively, and T representing the textual prompt
vectors.
As guiding signals, these positive and negative prompts
are compared with the image embeddings in CLIP’s shared
semantic space to enable semantically driven optimization.
They provide soft semantic supervision, helping the model
gradually adjust image features toward representations that
better align with human perception of high-quality images.
2) Sigmoid-Like Probabilistic Model: In our bi-level opti-
mization strategy, during the early and middle stages of
training, the enhanced images may remain in an “intermediate
state,” where their semantic distances to both positive and
negative prompts are relatively small. If prompt loss is
enforced too early, when image quality is still poor, it may
cause gradient instability and mislead the learning process. To
address this, the incorporation of prompt loss is governed by a
probabilistic model that simulates a sigmoid-shaped function
over the training duration. The decision to calculate prompt
loss at any given iteration is contingent upon the relative
progress of training, modeled as:
P(compute|iter, epoch) =
1
1 + exp(−k · (r −r0)),
(6)
where P(compute|iter, epoch) represents the probability of
computing the prompt loss at the current iteration and epoch, k
is a factor determining the steepness of the sigmoid function,
r is the relative position within the total training iterations,
calculated by r = e×T+t
E×T . Here, e is the current epoch (starting
from 0), t is the iteration index within the epoch, T is the
number of iterations per epoch, and E is the total number of
epochs. r0 is the relative position threshold, typically set to
0.9 to ensure a marked increase in the probability towards the
training’s conclusion.
This approach ensures the moderate and controlled intro-
duction of semantic prompt learning, signiﬁcantly reducing
training costs and enabling a smooth, dynamic transition from
structure-driven to semantics-driven optimization. It eﬀec-
tively integrates semantic guidance with the structural ﬁdelity
enforced by the structural similarity loss, achieving a harmo-
nious balance of detail preservation and illumination in the
generated images.
3) Integration With S3IM Loss: SSIM typically computes
structural similarity over ﬁxed sliding windows or the entire
image, which can easily overlook non-uniform variations in
local key regions, especially in cases involving localized
brightness enhancement or shadow removal. To complement
the prompt loss, we introduce the Stochastic Structural SIMi-
larity (S3IM) loss function [48], which extends the traditional
SSIM approach by randomly sampling local patches from
both the input and the NeRF-generated images to capture and
compare the non-local structural information contained within.
Formally, the loss is deﬁned as: LS 3IM(Θ, R)
=
1 −
S 3IM( ˜R, R), where S 3IM( ˜R, R) is computed as the average
of the SSIM values across M stochastic samples:
S 3IM( ˜R, R) = 1
M
M
X
m=1
S S IM(P(m)( ˜C), P(m)(C)).
(7)
Fig. 3. Illustration of the proposed bi-level optimization strategy. The yellow
path represents the optimization of only the NeRF network (lower-level)
parameters θ, while the green path corresponds to optimizing only the image
enhancement network (upper-level) parameters φ. The red path indicates the
joint optimization of φ and θ under proxy supervision, gradually converging
toward the optimal reﬁnement solution (red apple).
Here, P(m)( ˜C) and P(m)(C) denote the m-th stochastic
sampled patches from the NeRF-synthesized image and the
enhanced ground-truth image, respectively. The SSIM values
are computed using a predeﬁned kernel size K×K and a stride
size s, with the kernel size often set equal to the stride size to
ensure independent sampling of patches.
The ﬁnal loss for training the enhancement NeRF network
is the combination of the two losses:
Ltotal = Lprompt + α · LS 3IM,
(8)
where α is the weight to balance the magnitude of diﬀerent
loss terms which is set to 0.5 in our experiments.
C. Bi-Level Optimization With Enhancement and
Reconstruction
The key to 3D reconstruction in low-light conditions is to
establish information ﬂow and task coupling between image
enhancement and volume rendering to achieve semantic and
structural consistency. However, integrating image enhance-
ment with volume rendering in a uniﬁed pipeline remains
challenging. To overcome these limitations, we propose a bi-
level optimization framework, as illustrated in Figure 3, which
uniﬁes prompt-guided image enhancement and cue-guided
novel view generation based on NeRF into a two-layer struc-
ture. The upper-level task focuses on semantically constrained
optimization of the image enhancement network, while the
lower-level task corresponds to the 3D reconstruction net-
work. Unlike the traditional single-stage training strategy, by
introducing structural constraints in the parameter space, the
optimization of semantic cues in the upper layer can eﬀectively
inﬂuence the generation of new NeRF perspective in the lower
layer, and the quality of the 3D reconstruction also inﬂuences
the enhancement network in the upper layer through the proxy
supervision method, to achieve cyclic optimization of the
upper and lower layers and improve overall performance.
1) Optimistic
Bi-Level
Modeling:
The
conventional
approach inputs a set of image pairs IA and IB with normal
illumination and low illumination modules into an image
enhancement network Gφ for illumination processing to
generate an enhanced image IF = Gφ(IA, IB). Subsequently,
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 6 -->
2024
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
the enhanced image is used as the input image for new
perspective generation to train the NeRF network Rθ to
generate the new perspective rendering ˆI = Rθ(IF), but the
change in dimensionality aﬀects the eﬀectiveness of new
perspective generation, and in order to improve the relevance
of the image enhancement and 3D reconstruction tasks, we
introduce a bi-level optimization framework:
min
φ∈Φ F(φ, θ∗(φ)), s.t. θ∗(φ) = arg min
θ∈Θ f(φ, θ),
(9)
where f(φ, θ) denotes the volumetric rendering loss of the
lower NeRF network; F(φ, θ) denotes the semantic loss of the
upper image enhancement module, which is mainly composed
of the text prompt alignment loss ℓprompt and structural simi-
larity loss ℓS3IM; θ∗(φ) denotes the set of optimal parameters
of NeRF given the output of the enhanced image Gφ.
The key to the optimization framework is how to establish
the connection between the upper and lower layers of tasks.
We redeﬁne the upper-layer image enhancement task as a
“Proxy Supervision” strategy, in which the proxy supervision
signal is no longer directly derived from the pixel’s enhance-
ment criterion (e.g., luminance or contrast), but has the quality
of the lower-layer 3D reconstruction indirectly provided as
an index for evaluating the enhancement eﬀect. This model
improves the coupling between the upper and lower layers
of the task and enhances the semantic robustness and task
generalization ability during the training process.
The nested structure of the Bi-level optimization frame-
work itself, the parameter dependence and gradient circulation
between the upper and lower objective functions bring com-
plexity to the model, speciﬁcally, the upper objective function
F is a function of the lower solution θ, which itself implicitly
relies on the upper parameter φ. This introduces nonlinearity
into the optimization module, making gradient backpropa-
gation and convergence during training signiﬁcantly more
challenging.
2) Bi-Level Optimization Based on Loss Functions:
Although the Bi-level optimization theoretically separates the
upper layer semantic enhancement and the lower layer 3D
reconstruction tasks, its nested structure poses major diﬃcul-
ties in practical computation. In particular, when optimizing
the upper-layer parameter φ, the lower-layer optimal solution
θ∗(φ) relies on is obtained indirectly through the inner-layer
optimization process, which makes direct gradient computa-
tion diﬃcult.
We therefore use an approximate reconstruction scheme,
as outlined in Algorithm 1, to transform the original nested
structure into a diﬀerentiable process. The enhanced images
produced by the upper-level enhancement network are fed into
the NeRF model as input, directly inﬂuencing the reconstruc-
tion quality. Meanwhile, the novel-view images generated by
NeRF are compared with the enhanced images using structural
similarity (S3IM), and the resulting diﬀerences are used as a
backward feedback signal to guide the training of the image
enhancement network. In particular, we introduce a proxy
supervision term Lproxy that blends the original reconstruction
loss with a structurally adjusted version to provide a more
robust and informative gradient signal:
Lproxy ←β · LS3IM( ˆR, ˜R) + (1 −β) · LS3IM(A( ˆR), ˜R),
(10)
Algorithm 1 Approximate Reconstruction Scheme
1: Require: Image pairs (IA, IB); Target reconstruction ˜R;
Enhancement network Gφ; NeRF network Rθ; Loss func-
tions Lprompt, LS3IM; Learning rates ηφ, ηθ; Weights λ, β;
Iterations T
2: Initialize φ and θ
3: for t = 1 to T do
4:
% Enhancement stage
5:
IF ←Gφ(IA, IB), Compute Lprompt(IF)
6:
% Lower-level reconstruction stage
7:
ˆR ←Rθ(IF), Compute LS3IM( ˆR, ˜R)
8:
θ ←θ −ηθ · ∇θLS3IM( ˆR, ˜R)
9:
% Proxy supervision
10:
Lproxy ←β · LS3IM( ˆR, ˜R) + (1 −β) · LS3IM(A( ˆR), ˜R)
11:
% Upper-level enhancement update
12:
φ ←φ −ηφ · ∇φ

Lprompt(IF) + λ · Lproxy

13: end for
14: φ∗←φ, θ∗←θ
15: return φ∗, θ∗
where ˆR = Rθ(IF) is the reconstructed image, A(·) denotes a
structural alignment operation (e.g., brightness normalization,
histogram matching, or geometric alignment), and β ∈[0, 1]
balances the contribution between direct and adjusted sim-
ilarity. This formulation delivers indirect supervision with
structural adjustment, enhancing stability and convergence.
Notably, our scheme still leverages the original loss func-
tions, Lprompt and LS3IM, but the proxy supervision allows the
upper-level network to more eﬀectively aid the lower-level 3D
reconstruction task, while signiﬁcantly reducing computational
eﬀort.
Speciﬁcally, the lower task aims to learn the NeRF param-
eters θ by minimizing the structural reconstruction loss under
a ﬁxed augmented image IF = Gφ(IA, IB):
θ∗(φ) := arg min
θ∈Θ LS3IM(Rθ(Gφ(IA, IB)), ˜R),
(11)
where ˜R denotes the target image reconstruction result. The
upper-level goal focuses on optimizing the semantic represen-
tation of the augmented image to align it with the cue content:
min
φ∈Φ Lprompt(Gφ(IA, IB))
s.t.
θ = θ∗(φ).
(12)
In the speciﬁc training process, in order to avoid the
need to completely solve the inner optimal solution in each
iteration, we adopt the alternating optimization strategy. In
each iteration, θ and φ are updated alternately to approximate
the complete bilayer structure:
θ ←θ −ηθ · ∇θLS3IM(Rθ(IF), ˜R),
(13)
φ ←φ −ηφ · ∇φ

Lprompt(IF) + λ · Lproxy

.
(14)
The goal of the lower level task is to update the parameters
θ of the NeRF network to reconstruct the target result ˜R as
much as possible in terms of structural similarity, and the
goal of the upper level task is to update the parameters φ of
the image enhancement network to make the enhanced image
satisfy the semantic guidance as well as to help the lower level
3D reconstruction.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 7 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2025
IV. EXPERIMENTS
This section presents a comprehensive overview of the
experimental
framework,
including
data
preprocessing,
implementation details, baseline methods, and evaluation
metrics. Two representative datasets with varying resolutions
and
scene
complexities
were
selected
to
assess
the
generalization ability of the proposed method under diverse
real-world conditions. For baseline comparisons, we evaluated
several state-of-the-art NeRF enhancement methods, including
traditional low-light image enhancement algorithms and recent
models tailored for low-light scenarios, to emphasize the
strengths of our approach. We performed quantitative analyses
supported by visualizations of the rendered outputs. We also
evaluated the model’s performance in terms of training time,
memory consumption, and processing speed to provide a
comprehensive assessment of its eﬃciency and practicality.
Finally, a series of ablation studies were carried out to
thoroughly validate the eﬀectiveness of the proposed method
in generating high-quality, well-lit NeRF images under
challenging low-light conditions.
A. Datasets
1) LOM: We chose the LOM dataset, which contains 5 real-
life scenarios (“buu”, “chair”, “sofa”, “bike”, “shlub”). Each
scene contains 25 to 48 images. The original resolution of the
images is 3000 × 4000, and for computational convenience,
the dataset is downsampled to 375 × 500, and the real
viewpoint and angle information is generated by applying
COLMAP on the normal and low illumination images.
2) LLFF: The LLFF dataset consists of high-resolution
images at 4032 × 3024 resolution and includes four real-
world scenes (“fern”, “ﬂower”, “horns”, and “room”). Each
contains between 20 and 62 images. Since the original dataset
does not provide COLMAP reconstruction data, we re-ran
COLMAP under low-light conditions to obtain camera poses
for our experiments. For data splitting, each scene is divided
into training, validation, and testing sets in ﬁxed proportions.
B. Implementation Details
We evaluate the model on two datasets containing 9 real-
world scenes at high and low resolutions, covering a wide
range of lighting conditions. Our network training was exe-
cuted on 4 NVIDIA GeForce RTX 4090 GPUs, employing
the Adam optimizer for network adjustments. We performed
a diﬀerent number of training iterations and ﬁnally decided to
train for a total of 100 epochs, which is equivalent to 62,500
iterations, using a batch size of 1024 for each iteration. The
initial learning rate was set at 5e-4 and subjected to a cosine
decay adjustment every 2,500 iterations.
C. Baselines and Metrics
In this section, we carry out image generation quality
assessment using two high and low resolution datasets, LLFF
and LOM, to demonstrate the multi-view rendering capability
of our method under low-light conditions. Also, in choosing
the comparison method, we compare our method with two dif-
ferent classes of methods. First, we compare it with techniques
aimed at improving NeRF models, including some of the most
recent methods for applying NeRF in low-light environments,
such as Aleth-NeRF and AMENeRF.The second category
of methods involves preprocessing the low-light image by
applying enhancement methods to improve the quality of
the image, and then using the enhanced image for NeRF,
denoted as “* + NeRF.” This category includes nine popular
low-light enhancement methods.EnlightenGAN [49], HE [50],
IAT [51], LIME [52], PAIRILE [53], Retinexformer [54],
Retinexne [55], SCI [56] and Zero-DCE [57]. The metrics
evaluated in the experiments are PSNR, SSIM, and LPIPS. In
visualization, we focus on the eﬀectiveness of our method in
visual restoration and detail enhancement.
D. Qualitative Analysis
1) LOM: Some of our experimental results are shown in
Figure 4, which comprehensively demonstrates the advantages
of our method over other methods in terms of visual eﬀects. On
the whole, the images generated by our method are closer to
human perception. The second and sixth rows are the rendered
images of the “buu” scene, and we can see that AMENeRF
and Aleth-NeRF signiﬁcantly improve the brightness, but the
color recovery eﬀect is not as good as that of our method.
Locally, our method shows good detail recovery, such as
the “bike” scene in the ﬁrst row and the “shrub” scene
in the fourth row, whereas Aleth-NeRF signiﬁcantly improves
the image brightness but in many cases loses image details
due to overexposure. Although Aleth-NeRF can signiﬁcantly
improve the image brightness, in many cases, it will lose image
details due to overexposure. Our method is able to strike
a better balance between brightness enhancement and detail
retention. Our method adopts a bi-level optimization strategy
that utilizes both textual and structural information, which not
only eﬀectively enhances the overall image brightness, but also
maintains the detail realism well.
2) LLFF: The increase in resolution makes the exper-
imental computation higher, but we still ﬁx our training
epoch and iteration number, so it will be more diﬃcult to
accurately render high-resolution images. The results of some
of our experiments are shown in Figure 5. In the third and
seventh rows of the “horns” scene, we increased the diﬃculty
by choosing the hidden staircase balusters to compare the
details. Both Retinexformer and our method present clearer
balusters, but Retinexformer presents a more severe color loss
in the overall image, while our method maintains good color
recovery in both the overall and local areas at high resolution.
Our bi-level optimization method closely integrates the upper
and lower tasks during training and supervises each other,
allowing our method to achieve excellent results regardless
of high-resolution datasets or low-resolution datasets.
E. Quantitative Analysis
1) LOM: Our experimental results are shown in Table I,
where we selected two methods for NeRF enhancement in low-
light environments and nine 2D low-light image enhancement
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 8 -->
2026
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
Fig. 4. Visual analysis of NeRF enhancement in low-light conditions using the LOM dataset. Our method is compared with NeRF rendered images in low
light, NeRF rendered images in normal light and enhancement methods in low light.
methods, with the best metrics labeled in red and the second
best labeled by blue. From the experimental results, it can
be seen that our methods present excellent results, reﬂecting
the eﬀectiveness of our bi-level optimization methods, and
from other results, it can also be seen that the NeRF enhance-
ment methods speciﬁcally used for low-light environments are
signiﬁcantly better than the 2D low-light image enhancement
methods.
2) LLFF: Our results are shown in Table II. We used the
same enhancement method as the LOM dataset. In the high-
resolution dataset, our method shows better generalization
ability. The two-dimensional low-light image enhancement
method cannot establish information ﬂow and task coupling
between image enhancement and volume rendering under
high-resolution volume rendering conditions. In the “ﬂower”
scene, our PSNR index is 2.59 higher than the second one.
The proxy supervision mode of our bi-level optimization
framework establishes the connection between the upper and
lower layer tasks, improves the coupling between the upper
and lower layer tasks, and improves the task generalization
ability during training.
F. Evaluate and Analyze the Training Cost
Table III presents a comparison between our method and
AME-NeRF [16] and Aleth-NeRF [17] in terms of train-
ing time, memory usage, and other evaluation metrics. The
experimental setup is consistent with the description in the
implementation details, using the “bike” scene from the LOM
dataset. As shown in the table, although our method requires
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 9 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2027
Fig. 5. Visual analysis of NeRF enhancement in low-light conditions using the LLFF dataset. Our method is compared with NeRF rendered images in low
light, NeRF rendered images in normal light and enhancement methods in low light.
slightly more training time and GPU memory compared to the
baseline methods, it achieves signiﬁcantly better performance
in inference speed and evaluation metrics.
Bi-level optimization enables eﬃcient parameter updates
for both the upper and lower networks, achieving faster
convergence in the solution space compared to other methods
and increasing the likelihood of reaching a global optimum.
The CLIP-LIT module provides semantically aligned initial-
ization for the enhancement network. Since the prompt-based
supervision comes from a pretrained CLIP model, the infer-
ence is fast. The strong prior from the large model guides
the network toward accurate gradient descent, signiﬁcantly
reducing the number of iterations needed to escape random
or suboptimal regions in early training, thereby accelerating
overall convergence.
The S3IM loss ensures structural consistency between
NeRF-rendered
and
enhanced
images
through
a
non-
parametric, patch-based approach that is both lightweight and
eﬃcient. To further control computational cost and training
stability, the prompt loss is introduced gradually using a
sigmoid-shaped schedule, activating with low probability dur-
ing the early 90% of training. This allows the model to ﬁrst
focus on structure and only incorporate semantic supervision
once the image quality has improved.
G. Ablation Study
1) Evaluation on Important Modules: Table IV and V
show the quantitative analysis of the impact of using or not
using the two key modules of textual prompt constraint and
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 10 -->
2028
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
TABLE I
QUANTITATIVE COMPARISON OF THE EFFECTS OF LOW-LIGHT ENHANCEMENT ON THE LOM DATASET, COMPARING METRICS SUCH AS PSNR ↑, SSIM
↑, AND LPIPS ↓. THE BEST PERFORMANCE IS MARKED IN RED WHEREAS THE SECOND BEST ONE IS IN BLUE
TABLE II
QUANTITATIVE COMPARISON OF THE EFFECTS OF LOW-LIGHT ENHANCEMENT ON THE LLFF DATASET, COMPARING METRICS SUCH AS PSNR ↑, SSIM
↑, AND LPIPS ↓. THE BEST PERFORMANCE IS MARKED IN RED WHEREAS THE SECOND BEST ONE IS IN BLUE
TABLE III
COMPARISON OF TRAINING TIME, GPU MEMORY USAGE, INFERENCE
SPEED AND PERFORMANCE METRICS ON THE LOM DATASET (“BIKE”
SCENE)
structural similarity constraint on LOM and LLFF. We divided
the experiment into four parts, without using textual prompt
constraint and structural similarity constraint, using these two
loss functions separately, and using both loss functions. The
experimental results show that both loss functions are very
important and have a great impact on the indicators. In partic-
ular, the improvement in the structural similarity index (SSIM)
is particularly signiﬁcant, indicating that our method has a
signiﬁcant advantage in improving the consistency of image
structure. It is worth noting that the M3 model outperforms
M2 across most metrics in Table IV. The possible reason
is that 3D reconstruction under low illumination conditions
relies more on the structural information established by image
enhancement and 3D reconstruction based on text prompts.
2) Analysis of Training Strategy: Figure 6 shows the change
in visual quality of the model-generated images with diﬀerent
loss function settings, comparing the eﬀects of preprocessing
with CLIP-LIT only, with textual prompt constraints and struc-
tural similarity constraints, respectively, and with or without
a bi-level optimisation strategy. Our comparative analysis of
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 11 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2029
TABLE IV
QUANTITATIVE ABLATION ANALYSIS ON THE PROMPT LOSS AND S3IM LOSS WITHIN THE LOM DATASET. THE “PROMPT” DENOTES TEXTUAL PROMPT
CONSTRAINT Lprompt, AND “S3IM” DENOTES STRUCTURAL SIMILARITY CONSTRAINT LS 3IM
TABLE V
QUANTITATIVE ABLATION ANALYSIS ON THE PROMPT LOSS AND S3IM LOSS WITHIN LLFF DATASET
Fig. 6.
Visual comparison of ablation study on loss functions. The ﬁrst
row presents the images generated by diﬀerent models, while the second
row depicts their respective depth maps for direct correlation. The four
columns sequentially exhibit models utilizing: solely the CLIP-LIT module
(ﬁrst column), the CLIP-LIT module combined with S3IM Loss (second
column), the CLIP-LIT module augmented with prompt Loss (third column),
and our bi-level optimization (fourth column).
RGB images and depth images shows that when the model
uses both loss functions and chooses the bi-level optimisation
strategy, the generated images show the best results in terms
of colour reproduction, detail retention and spatial coherence.
As shown in Figure 7, we analyze the eﬀects of diﬀerent
training strategies on the enhancement eﬀect of NeRF. The
experimental results clearly show that after introducing the
preprocessing model and the bi-level optimization strategy,
NeRF signiﬁcantly improves the image quality in several
aspects. NeRF enhanced by the preprocessing model not only
improves the overall brightness but also shows a stronger
ability to maintain the details, especially in complex scenes
and low-light conditions, and the tiny details of the image
can be better recovered. By incorporating prompts, NeRF is
able to guide the model’s learning process more eﬀectively,
resulting in image features that are closer to high-quality image
standards. These prompts not only help the model to focus
on the key details of the image, but also guide the model to
TABLE VI
QUANTITATIVE ABLATION ANALYSIS ON THE BI-LEVEL OPTIMIZATION
STRATEGY WITHIN THE LOM AND LLFF DATASETS
focus on the areas related to the target features during image
synthesis, thus further optimizing the detail performance.
Table VI shows the quantitative analysis of the impact of
the bi-level optimization strategy on LOM and LLFF. In our
default setting, both the textual prompt constraint Lprompt and
the structural similarity constraint LS 3IM are enabled, as the
objective of the bi-level optimization is to jointly optimize
these two essential components. The results show that the bi-
level optimization strategy plays a crucial role in our method.
The objectives of image enhancement and NeRF recon-
struction are fundamentally diﬀerent: the former focuses on
semantic-aware quality, while the latter emphasizes geomet-
ric structural consistency. Direct joint training may lead to
conﬂicts, such as producing brighter images at the cost of
structural distortion, which negatively impacts NeRF recon-
struction. To address this, we adopt an alternating optimization
strategy: in each training round, we ﬁrst optimize the enhance-
ment network, then use the latest enhanced results to train the
NeRF. This forms a decoupled gradient ﬂow and a bidirec-
tional feedback loop, aligning the objectives of both tasks and
improving overall performance.
3) Experiments on Weights: In Figure 8, we explore the
eﬀect of diﬀerent loss function weights on the results in a
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 12 -->
2030
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
Fig. 7. Our approach is compared with NeRF-rendered images in low light, NeRF-rendered images with only 2D low-light enhancement using CLIP-LIT,
and enhancement methods after preprocessing the model and bi-level optimization strategy. (Note: The NeRF model’s output has been artiﬁcially brightened
for comparative clarity.)
Fig. 8. Visual comparison of ablation study on loss weights. The balance of
the two loss functions was explored by varying the weights of the Prompt
Loss and the S3IM Loss for quality assessment.
bi-level optimization strategy. By adjusting the weights of
the textual cue constraint and the S3IM (structural image
similarity) constraint, we observe the changes in the model
output and explore the equilibrium relationship between the
textual prompt loss (Prompt Loss) and the structural similarity
loss (S3IM Loss). When the weights of both loss functions are
low (e.g., Prompt = 0.2, S3IM = 0.1), the generated images
are low-brightness and have poor detail recovery. When only a
single loss weight is boosted (e.g., Prompt = 1.0, S3IM = 0.1)
although the brightness is improved, color distortion occurs
and structural consistency is lacking. The experimental results
show that when the Prompt constraint weight is set to 1 and
the S3IM constraint is set to 0.5, the best visual results are
obtained in terms of image brightness, color restoration and
detail resolution.
4) Impact of Textual Prompt Pairs: To delineate the inﬂu-
ence of positive and negative textual prompt pairs on our
Fig. 9.
Visual comparison of ablation study on textual prompts. The three
columns represent CLIP scores of training using only NEGATIVE prompts,
training using only POSITIVE prompts, and training using both types of
prompts, respectively.
approach, we present ablation studies in Figure 9. These exper-
iments were conducted using exclusively positive prompts,
exclusively negative prompts, and a combination of both. We
select two pairs of positive and negative prompts and use the
CLIP model to assess the validity of each prompt type. The
ﬁndings indicate that using both positive and negative prompts
together yields higher CLIP scores than using either type of
prompt in isolation. This underscores the signiﬁcant impact of
combining prompt polarities on both brightness enhancement
and texture preservation.
V. CONCLUSION
We propose a text-guided image enhancement method
that signiﬁcantly improves NeRF performance under low-
light conditions. By incorporating semantic prompts through
a visual-language model, our approach enhances bright-
ness while preserving detail, enabling more realistic NeRF
synthesis. A bi-level optimization framework with proxy
supervision jointly reﬁnes the enhancement and NeRF mod-
ules, achieving a balance between illumination consistency
and structural integrity. Experiments on the LOM and LLFF
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 13 -->
JU et al.: ILLUMINATION REFINEMENT VIA TEXTUAL CUES: A PROMPT-DRIVEN APPROACH
2031
datasets demonstrate state-of-the-art results. In future work, we
plan to explore temporal consistency for video-based NeRF
reconstruction under low-light conditions, ensuring smooth
appearance and geometry across frames. Additionally, optimiz-
ing our framework for real-time applications will be crucial for
deployment in dynamic or resource-constrained environments.
REFERENCES
[1]
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “NeRF: Representing scenes as neural radiance ﬁelds for
view synthesis,” Commun. ACM, vol. 65, no. 1, pp. 99–106, Jan. 2022.
[2]
Y. Ran et al., “CT-NeRF: Incremental optimization of neural radi-
ance ﬁeld and camera poses with complex trajectory,” IEEE Trans.
Circuits Syst. Video Technol., early access, Apr. 15, 2025, doi: 10.1109/
TCSVT.2025.3560900.
[3]
H. Qin, T. Yang, X. Zhou, D. Li, Y. Dai, and J. Li, “ECC-NeRF:
Anti-aliasing neural radiance ﬁelds with elliptic cone-casting for diverse
camera models,” IEEE Trans. Circuits Syst. Video Technol., early access,
May 28, 2025, doi: 10.1109/TCSVT.2025.3574474.
[4]
J. Liu, X. Fan, J. Jiang, R. Liu, and Z. Luo, “Learning a deep multi-
scale feature ensemble and an edge-attention guidance for image fusion,”
IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 1, pp. 105–119,
Jan. 2022.
[5]
Z. Huang, W. Hu, Z. Zhu, Q. Li, and H. Fang, “TMSF: Taylor expan-
sion approximation network with multi-stage feature representation for
optical ﬂow estimation,” Digit. Signal Process., vol. 162, Jul. 2025, Art.
no. 105157.
[6]
Z. Huang et al., “T2EA: Target-aware Taylor expansion approximation
network for infrared and visible image fusion,” IEEE Trans. Circuits
Syst. Video Technol., vol. 35, no. 5, pp. 4831–4845, Jan. 2025.
[7]
Z. Zhu, C. Huang, M. Xia, B. Xu, H. Fang, and Z. Huang, “RFRFlow:
Recurrent feature reﬁnement network for optical ﬂow estimation,” IEEE
Sensors J., vol. 23, no. 21, pp. 26357–26365, Nov. 2023.
[8]
Z. Zhu, M. Xia, B. Xu, Q. Li, and Z. Huang, “GTEA: Guided Taylor
expansion approximation network for optical ﬂow estimation,” IEEE
Sensors J., vol. 24, no. 4, pp. 5053–5061, Feb. 2024.
[9]
W. Wang, W. Yang, and J. Liu, “HLA-face: Joint high-low adaptation
for low light face detection,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2021, pp. 16190–16199.
[10] Y. Lei, X. Li, Z. Jiang, X. Ju, and J. Liu, “AEAM3D: Adverse
environment-adaptive monocular 3D object detection via feature extrac-
tion regularization,” in Proc. IEEE Int. Conf. Acoust., Speech Signal
Process. (ICASSP), Apr. 2024, pp. 4135–4139.
[11] X. Ju, X. Shang, X. Li, and B. Ren, “DART3D: Depth-aware robust
adversarial training for monocular 3D object detection,” Electron. Lett.,
vol. 61, no. 1, p. 70214, Jan. 2025.
[12] P. P. Srinivasan, B. Deng, X. Zhang, M. Tancik, B. Mildenhall, and
J. T. Barron, “NeRV: Neural reﬂectance and visibility ﬁelds for relighting
and view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), Jun. 2021, pp. 7495–7504.
[13] Q. Yan et al., “Eﬃcient image enhancement with a diﬀusion-based
frequency prior,” IEEE Trans. Circuits Syst. Video Technol., early access,
Mar. 7, 2025, doi: 10.1109/TCSVT.2025.3549351.
[14] M. Liu, Y. Cui, W. Ren, J. Zhou, and A. C. Knoll, “LIEDNet: A
lightweight network for low-light enhancement and deblurring,” IEEE
Trans. Circuits Syst. Video Technol., vol. 35, no. 7, pp. 6602–6615, Jul.
2025.
[15] J. Hou, Z. Zhu, J. Hou, H. Liu, H. Zeng, and H. Yuan, “Global
structure-aware diﬀusion process for low-light image enhancement,”
2023, arXiv:2310.17577.
[16] Y. Zou, X. R. Li, Z. Jiang, and J. Liu, “Enhancing neural radiance ﬁelds
with adaptive multi-exposure fusion: A bilevel optimization approach for
novel view synthesis,” in Proc. AAAI Conf. Artif. Intell., 2024, vol. 38,
no. 7, pp. 7882–7890.
[17] Z. Cui, L. Gu, X. Sun, X. Ma, Y. Qiao, and T. Harada, “Aleth-NeRF:
Illumination adaptive NeRF with concealing ﬁeld assumption,” in Proc.
AAAI Conf. Artif. Intell., 2024, vol. 38, no. 2, pp. 1435–1444.
[18] H. Zhou, W. Dong, and J. Chen, “LITA-GS: Illumination-agnostic novel
view synthesis via reference-free 3D Gaussian splatting and physi-
cal priors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2025, pp. 21580–21589.
[19] B. Mildenhall et al., “Local light ﬁeld fusion: Practical view synthesis
with prescriptive sampling guidelines,” ACM Trans. Graph., vol. 38,
no. 4, pp. 1–14, Aug. 2019.
[20] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “PixelNeRF: Neural radiance
ﬁelds from one or few images,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2021, pp. 4578–4587.
[21] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger,
and N. Radwan, “RegNeRF: Regularizing neural radiance ﬁelds for view
synthesis from sparse inputs,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2022, pp. 5480–5490.
[22] A. Jain, M. Tancik, and P. Abbeel, “Putting NeRF on a diet: Semantically
consistent few-shot view synthesis,” in Proc. IEEE/CVF Int. Conf.
Comput. Vis. (ICCV), Oct. 2021, pp. 5885–5894.
[23] L. Ma et al., “Deblur-NeRF: Neural radiance ﬁelds from blurry images,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
2022, pp. 12851–12860.
[24] H. Wang, X. Xu, K. Xu, and R. W. H. Lau, “Lighting up NeRF via
unsupervised decomposition and enhancement,” in Proc. IEEE/CVF Int.
Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 12632–12641.
[25] B. Mildenhall, P. Hedman, R. Martin-Brualla, P. P. Srinivasan, and
J. T. Barron, “NeRF in the dark: High dynamic range view synthesis
from noisy raw images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., May 2022, pp. 16190–16199.
[26] X. Li et al., “DifIISR: A diﬀusion model with gradient guidance for
infrared image super-resolution,” 2025, arXiv:2503.01187.
[27] X. Li et al., “From text to pixels: A context-aware semantic synergy
solution for infrared and visible image fusion,” 2023, arXiv:2401.00421.
[28] J. Liu et al., “PromptFusion: Harmonized semantic prompt learning for
infrared and visible image fusion,” IEEE/CAA J. Autom. Sinica, vol. 12,
no. 3, pp. 502–515, Mar. 2025.
[29] X. R. Li et al., “Contourlet residual for prompt learning enhanced
infrared image super-resolution,” in Proc. Eur. Conf. Comput. Vis. Cham,
Switzerland: Springer, 2024, pp. 270–288.
[30] W. Yang, S. Wang, Y. Fang, Y. Wang, and J. Liu, “Band representation-
based semi-supervised low-light image enhancement: Bridging the gap
between signal ﬁdelity and perceptual quality,” IEEE Trans. Image
Process., vol. 30, pp. 3461–3473, 2021.
[31] L. Zhu, W. Yang, B. Chen, F. Lu, and S. Wang, “Enlightening low-light
images with dynamic guidance for context enrichment,” IEEE Trans.
Circuits Syst. Video Technol., vol. 32, no. 8, pp. 5068–5079, Aug. 2022.
[32] Z. Cui, X. Chu, and T. Harada, “Luminance-GS: Adapting 3D Gaussian
splatting to challenging lighting conditions with view-adaptive curve
adjustment,” 2025, arXiv:2504.01503.
[33] R. C. Gonzales and B. A. Fittes, “Gray-level transformations for inter-
active image enhancement,” Mechanism Mach. Theory, vol. 12, no. 1,
pp. 111–122, Jan. 1977.
[34] S. M. Pizer et al., “Adaptive histogram equalization and its variations,”
Comput. Vis., vol. 39, no. 3, pp. 355–368, 1987.
[35] D. J. Jobson, “Retinex processing for automatic image enhancement,”
J. Electron. Imag., vol. 13, no. 1, pp. 100–110, Jan. 2004.
[36] K. G. Lore, A. Akintayo, and S. Sarkar, “LLNet: A deep autoencoder
approach to natural low-light image enhancement,” Pattern Recognit.,
vol. 61, pp. 650–662, Jan. 2017.
[37] Y. Wu et al., “Learning semantic-aware knowledge guidance for low-
light image enhancement,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2023, pp. 1662–1671.
[38] S. Yang, M. Ding, Y. Wu, Z. Li, and J. Zhang, “Implicit neural
representation for cooperative low-light image enhancement,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 12872–12881.
[39] A. Radford et al., “Learning transferable visual models from natu-
ral language supervision,” in Proc. Int. Conf. Mach. Learn., 2021,
pp. 8748–8763.
[40] D. Liang, J. Xie, Z. Zou, X. Ye, W. Xu, and X. Bai, “CrowdClip: Unsu-
pervised crowd counting via vision-language model,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., Jun. 2023, pp. 2893–2903.
[41] R. Liu, J. Huang, G. Li, J. Feng, X. Wu, and T. H. Li, “Revisiting tem-
poral modeling for clip-based image-to-video knowledge transferring,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
2023, pp. 6555–6564.
[42] C. Zhou, C. C. Loy, and B. Dai, “Extract free dense labels from CLIP,”
in Proc. Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2022,
pp. 696–712.
[43] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for vision-
language models,” Int. J. Comput. Vis., vol. 130, no. 9, pp. 2337–2348,
2022.
[44] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Conditional prompt learning
for vision-language models,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jun. 2022, pp. 16816–16825.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 14 -->
2032
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
[45] Z. Liang, C. Li, S. Zhou, R. Feng, and C. C. Loy, “Iterative prompt learn-
ing for unsupervised backlit image enhancement,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 8094–8103.
[46] F. Yang et al., “Dynamic prompt learning: Addressing cross-attention
leakage for text-based image editing,” in Proc. Adv. Neural Inf. Process.
Syst., vol. 36, 2023, pp. 26291–26303.
[47] V. Potlapalli, S. Waqas Zamir, S. Khan, and F. Shahbaz Khan,
“PromptIR: Prompting for all-in-one blind image restoration,” 2023,
arXiv:2306.13090.
[48] Z. Xie et al., “S3IM: Stochastic structural SIMilarity and its unrea-
sonable eﬀectiveness for neural ﬁelds,” in Proc. IEEE/CVF Int. Conf.
Comput. Vis. (ICCV), Oct. 2023, pp. 17978–17988.
[49] Y. Jiang et al., “EnlightenGAN: Deep light enhancement without paired
supervision,” IEEE Trans. Image Process., vol. 30, pp. 2340–2349,
2021.
[50] O. Patel, Y. P. S. Maravi, and S. Sharma, “A comparative study of his-
togram equalization based image enhancement techniques for brightness
preservation and contrast enhancement,” 2013, arXiv:1311.4033.
[51] Z. Cui et al., “You only need 90K parameters to adapt light: A light
weight transformer for image enhancement and exposure correction,”
2022, arXiv:2205.14871.
[52] X. Guo, Y. Li, and H. Ling, “LIME: Low-light image enhancement
via illumination map estimation,” IEEE Trans. Image Process., vol. 26,
no. 2, pp. 982–993, Feb. 2017.
[53] Z. Fu, Y. Yang, X. Tu, Y. Huang, X. Ding, and K.-K. Ma, “Learning
a simple low-light image enhancer from paired low-light instances,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2023,
pp. 22252–22261.
[54] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang,
“Retinexformer: One-stage retinex-based transformer for low-light
image enhancement,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Oct.
2023, pp. 12504–12513.
[55] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition
for low-light enhancement,” 2018, arXiv:1808.04560.
[56] L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward fast, ﬂexible, and
robust low-light image enhancement,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 5637–5646.
[57] C. Guo et al., “Zero-reference deep curve estimation for low-light image
enhancement,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2020, pp. 1780–1789.
Xinrui Ju received the B.E. degree in software
engineering from Dalian University of Technology,
Dalian, China, in 2025. He is currently pursuing the
M.Phil. degree with the Department of Computer
Science, City University of Hong Kong. His research
interests include low-level vision and 3D reconstruc-
tion.
Yang
Zou
received the master’s degree from
the School of Computer Science, The University
of Sydney, Australia, in 2024. He is currently
pursuing the Ph.D. degree with Northwestern Poly-
technical
University.
He
is
aﬃliated
with
the
National
Engineering
Laboratory
for
Integrated
Aero-Space–Ground–Ocean Big Data Application
Technology, Northwestern Polytechnical University.
His research interests include computer vision, low-
level vision, image fusion, and image enhancement.
Xingyuan Li received the bachelor’s degree from
the School of Software, Dalian University of Tech-
nology, Dalian, China, in 2022, where he is currently
pursuing the master’s degree. He is aﬃliated with the
Key Laboratory for Ubiquitous Network and Service
Software of Liaoning Province, Dalian University of
Technology. His research interests include computer
vision, low-level vision, image fusion, and image
enhancement.
Zirui Wang is currently pursuing the bachelor’s
degree with the School of International Informa-
tion and Software, Dalian University of Technology,
Dalian, China. His major is digital media and
technology. His research interests include low-level
vision and image fusion.
Jun Ma is currently pursuing the bachelor’s degree
with the School of International Information and
Software, Dalian University of Technology, Dalian,
China. His major is software engineering. His
research interests include low-level vision and image
fusion.
Zhiying Jiang received the B.E. degree in soft-
ware engineering from Dalian Maritime University,
China, in 2017, and the M.S. and Ph.D. degrees
in software engineering from Dalian University of
Technology, China, in 2020 and 2024, respectively.
She is currently with the College of Information
Science and Technology, Dalian Maritime Univer-
sity. Her research interests include computer vision,
image restoration, and image stitching.
Jinyuan Liu (Member, IEEE) received the M.S.
degree in computer science from Dalian Univer-
sity, Dalian, China, in 2018, and the Ph.D. degree
in software engineering from Dalian University of
Technology, Dalian, in 2022. He is currently a Post-
Doctoral Fellow with the School of Mechanical
Engineering, Dalian University of Technology. He
is also aﬃliated with the Key Laboratory for Ubiq-
uitous Network and Service Software of Liaoning
Province. His research interests include computer
vision, image fusion, and deep learning.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on April 08,2026 at 02:37:12 UTC from IEEE Xplore.  Restrictions apply.
