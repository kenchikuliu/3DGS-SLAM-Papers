# Intellectual Property Protection for 3D Gaussian Splatting Assets: A Survey

Longjie Zhao1 , Ziming Hong1 , Jiaxin Huang3 , Runnan Chen1 ,

Mingming Gong2,3 and Tongliang Liu1,3,â

1Sydney AI Centre, The University of Sydney, 2The University of Melbourne

3Mohamed bin Zayed University of Artificial Intelligence

lzha0538@uni.sydney.edu.au, hoongzm@gmail.com, jiaxin.huang@mbzuai.ac.ae, mingming.gong@unimelb.edu.au, {runnan.chen,tongliang.liu}@sydney.edu.au

## Abstract

3D Gaussian Splatting (3DGS) has become a mainstream representation for real-time 3D scene synthesis, enabling applications in virtual and augmented reality, robotics, and 3D content creation. Its rising commercial value and explicit parametric structure raise emerging intellectual property (IP) protection concerns, prompting a surge of research on 3DGS IP protection. However, current progress remains fragmented, lacking a unified view of the underlying mechanisms, protection paradigms, and robustness challenges. To address this gap, we present the first systematic survey on 3DGS IP protection and introduce a bottom-up framework that examines (i) underlying Gaussian-based perturbation mechanisms, (ii) passive and active protection paradigms, and (iii) robustness threats under emerging generative AI era, revealing gaps in technical foundations and robustness characterization and indicating opportunities for deeper investigation. Finally, we outline six research directions across robustness, efficiency, and protection paradigms, offering a roadmap toward reliable and trustworthy IP protection for 3DGS assets. A collection of relevant papers is summarized and will be continuously updated at https://github.com/tmllab/ Awesome-3DGS-IP-Protection.

## 1 Introduction

3D Gaussian Splatting (3DGS) [1â3] revolutionizes novel view synthesis by representing scenes or objects as collections of anisotropic 3D Gaussians with learnable attributes, offering superior rendering quality and real-time efficiency compared to existing 3D representations (e.g. Neural Radiance Fields (NeRF) [4]). These advantages have driven transformative applications across virtual/augmented reality [5], robotic perception [6â8], AI-generated 3D content [9, 10], and immersive gaming experiences [11]. Its widespread adoption has created a burgeoning ecosystem of high-value digital assets.

However, the rapid adoption of 3DGS also introduces critical intellectual property (IP) protection challenges, where IP for digital 3D assets refers to the exclusive rights governing the ownership, control, and authorized use of creative or commercially valuable content (i.e., the legal entitlement to determine how such assets are accessed, modified, and distributed). Given both the high commercial value and the explicit parametric formulation of 3DGS assets, attackers are provided with high incentives and relative low technical barriers for IP misuse, including but not limited to: (i) unauthorized extraction, copying, and redistribution of asset parameters or rendered outputs [12, 13], and (ii) malicious generative editing that produces derivative assets while evading provenance or licensing [14, 15]. Although existing IP protection techniques target 2D images or other 3D representations [16â20], they cannot be directly applied to 3DGS: 2D protections typically vanish during 3DGS reconstruction, and 3DGS relies on Gaussian primitives that differ fundamentally from implicit NeRF or mesh-based representations. Accordingly, this highlights the urgent need for effective IP protection methods tailored to 3DGS.

Researches on 3DGS IP protection has grown rapidly since 2024, with primary efforts focusing on 3DGS watermarking [12, 21] and 3DGS steganography [22, 23], and more recent studies extending to 3DGS tampering localization [24] and 3DGS editing safeguard [15]. Despite these progress, three critical gaps prevent the field from advancing systematically:

â¢ Fragmented research landscape. Existing works span multiple IP protection tasks, but lack a unified taxonomy to organize protection mechanisms, application scenarios, and their relationships. This prevents researchers from understanding the broader technical landscape and identifying optimal strategies under different security requirements.

â¢ Lack of underlying technical analysis. Across different 3DGS IP protection tasks, the techniques primarily rely on invisible 3D Gaussian-based perturbations. However, no prior work has systematically summarized the technical foundations or analyzed how they contribute to invisibility, protection capacity, and robustness.

â¢ Incomplete robustness characterization. While certain approaches assess robustness against conventional 2D and 3D distortions [12, 21], the AI-Generated Content (AIGC) era [25â30] introduces more sophisticated threats driven by the most advanced generative models, such as generative purification attacks [31], as well as the impact of generative editing on the integrity of protected assets [15, 32], which are largely unexplored.

<!-- image-->  
(a) Gaussian-based perturbations

<!-- image-->  
(b) 3DGS IP protection tasks

<!-- image-->  
(c) Robustness  
Figure 1: We introduce a bottom-up framework that summarize 3DGS IP protection from (a) underlying mechanisms of Gaussian-based perturbation (Section 3), through (b) 3DGS IP protection tasks (Sections 4 and 5), to (c) emerging robustness threats in AIGC era (Section 6).

## 2 Preliminary

To bridge this gap, we present the first systematic survey on 3DGS IP protection. We introduce a bottom-up framework (as shown in Figure 1) that progresses from (i) Gaussianbased perturbation, through (ii) 3DGS IP protection tasks, to (iii) emerging robustness threats in the AIGC era, and we categorize 24 existing methods under this framework in Table 1. Specifically, we first focus on underlying mechanisms of Gaussian-based perturbation (Section 3), where we summarize existing techniques along three dimensions: attribute selection mechanism, distribution strategy, and perturbation injection pipelines. For 3DGS IP protection tasks, we distinguish existing methods between passive protection and active protection strategies depending on whether the goal is posthoc tracing or proactive usability restriction, and provide a structured summary of both in Section 4 and Section 5, respectively. For emerging robustness threats (Section 6), we begin with conventional 2D and 3D distortions and progressively expand to intentional generative purification attacks, and generative editing which can unintentional eliminate protective perturbations. Finally, in Section 7, we outline six future research directions and their associated challenges for 3DGS IP protection across three key dimensions: robustness, efficiency, and protection paradigms. We believe these directions will further advance robust 3DGS IP protection and support trustworthy deployment of 3DGS assets.

Our contributions are summarized as follows:

â¢ Systematic survey. We present the first systematic survey of 3DGS IP protection, structuring existing methods into passive and active protection paradigms.

â¢ Underlying technique analysis. We analyze Gaussianbased perturbation, which serves as the primarily shared underlying technique across different 3DGS IP protection methods, spanning attribute selection mechanism, distribution strategy, and perturbation injection pipelines.

â¢ Robustness issues. We highlight critical robustness risks, ranging from conventional 2D and 3D distortions to AIGCdriven generative purification and editing.

â¢ Future roadmap. We outline six actionable research directions for securing 3DGS assets in the AIGC era.

3D Gaussian Splatting (3DGS). 3DGS [1, 33] represents a 3D scene or object as a collection of anisotropic 3D Gaussians $G ~ = ~ \{ g _ { 1 } , . . . , g _ { N } \}$ Each Gaussian primitive gi is parameterized by a set of learnable attributes $\Theta _ { i } \ =$ $\{ \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , \mathbf { c } _ { i } \}$ , where $\mu _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ denotes the position, $\Sigma _ { i }$ is the covariance matrix, $\alpha _ { i } ~ \in ~ [ 0 , 1 ]$ is the opacity, and ci represents the view-dependent color modeled by Spherical Harmonics (SH) coefficients. During rendering, these 3D Gaussians are projected into 2D image space. For a given pixel p, the accumulated color C(p) is computed via Î±-blending of the N sorted Gaussians overlapping with the pixel: $\begin{array} { r } { \mathbf { C } ( p ) \ = \ \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } ) } \end{array}$ , where $\alpha _ { i } ^ { \prime }$ is the effective opacity computed by multiplying $\alpha _ { i }$ with the 2D Gaussian evaluation at pixel p. The differentiable nature of this rasterization pipeline allows for the back-propagation of gradients from 2D loss functions to 3D parameters Î, which forms the mathematical basis. Driven by its real-time rendering and high visual fidelity, 3DGS has rapidly become a dominant paradigm for 3D scene representation [5, 34â36], thereby amplifying the need for IP protection mechanisms.

Taxonomy for 3DGS IP protection. IP for digital 3D assets refers to the exclusive rights governing the ownership, control, and authorized use of creative or commercially valuable content (i.e., the legal entitlement to determine how such assets are accessed, modified, and distributed) [26, 37]. As shown in Figure 1(b), with the rise of 3DGS, corresponding IP protection techniques have introduced diverse objectives and mechanisms, making it necessary to organize them within a unified taxonomy. In this survey, we categorize existing methods based on their protective functionalities into two primary streams:

â¢ Passive 3DGS IP protection enables post-hoc verification and traceability of IP without actively restricting 3DGS asset functionality. These methods embed verifiable perturbations into the 3DGS asset to establish ownership or provenance, and to detect unauthorized tampering [38â41]. Existing passive approaches include: (i) 3DGS watermarking [12], which embeds robust identifiers for ownership verification; (ii) 3DGS steganography [42], which conceals arbitrary payloads for covert communication or provenance; and (iii) 3DGS tampering localization [24], which reveals spatial regions undergoing malicious modification.

â¢ Active 3DGS IP protection provides IP protection by proactively restricting the usability of 3DGS assets for unauthorized downstream tasks $[ 4 3 - 4 6 ]$ . Rather than tracing assets, these methods actively restrict unauthorized manipulation to enforce usage compliance and prevent misuse [47]. Existing approaches include 3DGS editing safeguard [15], which protects 3DGS assets against instruction-driven generative editing.

General 3DGS IP protection objective. Regardless of the specific protection goals, we propose a general framework of 3DGS IP protection. Let G denote a dataset of original 3DGS assets, and $G \sim \mathcal G$ is a sampled asset. We denote the protected version as ${ \tilde { G } } = f _ { \theta } ( G )$ , where $f _ { \theta }$ denotes a protection operator parameterized by $\theta \ ( \mathrm { e . g . }$ , learnable network weights or optimized Gaussian parameters). The general objective can be formulated as follows:

$$
\begin{array} { r l } & { \underset { \theta } { \mathrm { m a x } } \underbrace { \mathbb { E } _ { G \sim \mathcal { G } } \left[ s ( f _ { \theta } ( G ) ; \tau ) \right] } _ { \mathrm { P r o t e c t i v e : a p a c i t y } \ \mathcal { L } _ { p } ( \theta ) } + \lambda \underbrace { \mathbb { E } _ { G \sim \mathcal { G } , T \sim \mathcal { T } _ { \mathrm { r o b } } } \left[ s ( T ( f _ { \theta } ( G ) ) ; \tau ) \right] } _ { \mathrm { R o b u s t n e s s } \ \mathcal { L } _ { r } ( \theta ) } } \\ & { \mathrm { s . t . } \ \underbrace { \mathbb { E } _ { G \sim \mathcal { G } , v \sim \mathcal { V } } \left[ d ( R _ { v } ( G ) , R _ { v } ( f _ { \theta } ( G ) ) ) \right] } _ { \mathrm { I m p e r c e p t i b i l i t y } \ \mathcal { C } _ { v } ( \theta ) } \leq \epsilon , } \end{array}
$$

which relies on three fundamental properties:

â¢ Imperceptibility $\mathcal { C } _ { v } ( \theta ) \leq \epsilon$ enforces that the protected asset $R _ { v } ( { \dot { P } } _ { \theta } ( G ) )$ remains perceptually close to the original $R _ { v } ( G )$ , where $d ( \cdot , \cdot )$ measures perceptual distance and Ïµ is a distortion budget. This ensures that IP protection does not degrade the photorealism or visual fidelity of 3DGS assets.

â¢ Protective capacity $\mathcal { L } _ { p } ( \theta )$ evaluates the direct protection performance for a given task Ï via a task-specific scoring function $s ( \cdot ; \tau )$ , where higher scores correspond to stronger task compliance. Notably, different protection tasks may entail different operational objectives (Section 4 and 5).

â¢ Robustness ${ \mathcal { L } } _ { r } ( \theta )$ characterizes how well the protection performance survives under transformations $T ~ \sim ~ \tau _ { \mathrm { r o b } } ,$ where $\mathcal { T } _ { \mathrm { r o b } }$ may include conventional distortions as well as generative purification or editing, applied either to 2D rendered views or directly to the 3D asset space (Section 6).

## 3 Gaussian-based Perturbations

A underlying technique shared between 3DGS IP protection methods is to embed Gaussian-based perturbations, as shown in Figure 1(a). Different perturbation strategies inherently shape the trade-off among protection capacity, visual fidelity, and robustness. For clarity, we categorize Gaussianbased perturbations along three dimensions: (i) attribute selection mechanism, (ii) distribution strategy, and (iii) injection pipelines.

## 3.1 Attribute Selection Mechanism

The most fundamental strategy lies in deciding which Gaussian attributes are permitted to carry perturbations. Gaussian parameters are inherently heterogeneous because different attributes exhibit distinct perceptual sensitivity, embedding capacity, and robustness, resulting in different trade-offs when chosen as perturbation carriers. Existing strategies can be categorized into SH-only perturbation, hybrid-attributes perturbation, and auxiliary attribute coupling.

Spherical Harmonics perturbation. SH coefficients are a popular choice for perturbation embedding in 3DGS [21â23, 50, 54], as modifying SH only affects color/appearance while preserving the underlying Gaussian geometry (e.g., position and covariance), thus maintaining high visual fidelity. To avoid obvious color distortions [58], existing methods usually introduce SH offsets as residual signals [21], partition higher-order SH bands [54], or replace SH with coupled latent features [22]. However, the high-dimensional and complex structure of SH coefficients leads to unstable optimization [58]. Moreover, recent studies report that SH perturbation suffers from high statistical detectability [58] and even insufficient protective capability against editing [15].

Hybrid-attributes perturbation. More methods adopt hybrid-attribute perturbation, where perturbations are injected not only into SH coefficients but also into geometryrelated Gaussian attributes (e.g., position, covariance, opacity) [15, 62]. This strategy enhances protection capability by reducing reliance on single attribute. However, geometry attributes are highly sensitive to perturbations, e.g., even minor deviations in position can significantly degrade rendering quality [21, 57], while perturbing covariance parameters often introduces noticeable artifacts, particularly near object boundaries [21, 50]. As a result, although hybridattribute perturbation can strengthen protection and avoid isolated color distortion, it may also incur broader visual degradation due to geometric sensitivity, and typically requires more complex optimization to balance visual quality and protective capability on challenging protection tasks [15, 62].

Auxiliary attribute coupling. A small number of methods attach auxiliary protection attributes to the original Gaussians, i.e., they externally append new attributes dedicated to carrying protective perturbations, rather than modifying any native Gaussian attributes [22, 42]. Since these auxiliary channels do not interfere with the original parameters, they can fully preserve visual quality and support highcapacity embedding. However, since these auxiliary channels are not coupled with semantically meaningful attributes or rendering-critical parameters, adversaries may prune or sanitize them without degrading visual quality, thus limiting their practical robustness.

## 3.2 Distribution Strategy

The distribution strategy determines how perturbations are distributed across the Gaussian primitives of a 3DGS asset. Several methods selectively perturb a subset of Gaussians with specific distributions (referred to as local strategies), typically using Fourier frequency or uncertainty as guidance for subset selection. In contrast, methods that do not impose any explicit selection constraints on Gaussian primitives, and instead apply perturbations across the entire asset, are referred to as a global strategy.

<table><tr><td rowspan="3">Paper Information</td><td rowspan="2" colspan="2"></td><td colspan="3">(a) Gaussian-based Perturbations</td><td colspan="2">(b) Task-specific</td><td rowspan="2"></td><td colspan="5">(c) Robustness</td></tr><tr><td>Attribute</td><td>Distribution</td><td>Inject</td><td>M-Dim</td><td>P-Dim</td><td>Aug PEP</td><td>2D Attacks</td><td></td><td></td><td>3D Attacks</td></tr><tr><td></td><td>Î¼Î£Î±c</td><td></td><td>Be J0T</td><td>deb E</td><td></td><td>3</td><td></td><td>US!or doD Bru</td><td>Stte pooo Roet</td><td>qmn ced Gee eunn</td><td>Us!or Srn Roe</td><td>Ste dn cne Ced</td></tr><tr><td>Method</td><td>Venue</td><td></td><td>qoD a</td><td>u</td><td>Gen 3DGS Watermarking</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GaussianMarker [12]</td><td>NeurIPS&#x27;24</td><td>â â</td><td></td><td>â â</td><td>â</td><td></td><td></td><td></td><td>â</td><td>âââ â</td><td></td><td>â</td><td></td></tr><tr><td>WATER-GS [48]</td><td>arXiv&#x27;24</td><td>â â â</td><td>â â</td><td></td><td>â</td><td>â â</td><td>â â</td><td>â â â</td><td></td><td></td><td>â</td><td>â \ Â¸</td><td></td></tr><tr><td>3D-GSW [49]</td><td>CVPR&#x27;25</td><td>â Â¸</td><td>â</td><td>â</td><td>â</td><td>â</td><td></td><td></td><td>â â</td><td>â â â</td><td>â</td><td></td><td>â</td></tr><tr><td>GuardSplat [21]</td><td>CVPR&#x27;25</td><td></td><td>â</td><td></td><td>â</td><td></td><td>&gt;&gt;</td><td>â</td><td>â â â</td><td>â â â</td><td>â</td><td>Â¸â</td><td></td></tr><tr><td>MarkSplatter [50]</td><td>ACMMM&#x27;25</td><td></td><td>â</td><td>â</td><td>â</td><td></td><td>â â</td><td>â</td><td>â â â â</td><td>â â â</td><td>â</td><td>âââÂ¸</td><td></td></tr><tr><td>MantleMark [51]</td><td>TIFS&#x27;25</td><td>â ââ</td><td>â</td><td>â â</td><td>â</td><td>â</td><td>â</td><td></td><td>â â â</td><td>â â</td><td>â</td><td></td><td></td></tr><tr><td>NGS-Marker [52]</td><td>OpenReview&#x27;25</td><td>â â â</td><td>â</td><td>â</td><td>â</td><td>â â</td><td>â</td><td></td><td></td><td></td><td>â</td><td></td><td>â â â</td></tr><tr><td>Mark3DGS [53]</td><td>OpenReview&#x27;25</td><td></td><td>â</td><td>â â</td><td>â</td><td></td><td>â</td><td>â</td><td>â â</td><td>Â¸Â¸Â¸</td><td></td><td>â â â â</td><td>â</td></tr><tr><td>X-SG2s [54]</td><td>arXiv&#x27;25</td><td></td><td>â</td><td>â</td><td>â</td><td>â â â</td><td>â</td><td></td><td></td><td></td><td></td><td>â â</td><td></td></tr><tr><td>CompMarkGS [55]</td><td>arXiv&#x27;25</td><td>â</td><td>â â â</td><td>â</td><td>â</td><td></td><td>â</td><td>Â¸Â¸</td><td></td><td>â â â â</td><td></td><td>â</td><td>â</td></tr><tr><td>GaussianSeal [56]</td><td>arXiv&#x27;25</td><td>â</td><td>â</td><td>â â</td><td></td><td></td><td>â</td><td></td><td>â â â</td><td></td><td>â</td><td>â â</td><td>â</td></tr><tr><td>GS-Marker [57]</td><td></td><td>â â</td><td>â</td><td>â</td><td></td><td></td><td>â</td><td></td><td>â</td><td>â â â</td><td>â</td><td>&gt;&gt;</td><td>â</td></tr><tr><td></td><td>arXiv&#x27;25</td><td>â â</td><td>â</td><td>â</td><td>â â</td><td></td><td>â</td><td>ââ</td><td>â â</td><td></td><td></td><td></td><td></td></tr><tr><td>RDSplat [14]</td><td>arXiv&#x27;25</td><td>Â¸Â¸ â</td><td>â</td><td>ââ</td><td></td><td></td><td></td><td>â</td><td>âÂ¸</td><td></td><td>â</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>3DGS Steganography</td><td>â â</td><td>â</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-Hider [22] StegaGaussian [23]</td><td>NeurIPS&#x27;24 KBS&#x27;25</td><td></td><td>â â</td><td>â â â</td><td>â â</td><td>â</td><td>â</td><td>â</td><td></td><td>ââââÂ¸Â¸</td><td></td><td></td><td></td></tr><tr><td>KeySS [58]</td><td>arXiv&#x27;25</td><td>â</td><td>â</td><td>â</td><td>â</td><td></td><td></td><td></td><td></td><td></td><td></td><td>~Â¸</td><td></td></tr><tr><td>Hide A Bit [59]</td><td>SIGGRAPH&#x27;25</td><td>â</td><td>â â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InstantSplamp [60]</td><td>ICLR&#x27;25</td><td>â â</td><td>â â</td><td>â</td><td></td><td>â</td><td>â</td><td></td><td>âââ</td><td></td><td></td><td></td><td></td></tr><tr><td>SecureGS [42]</td><td>ICLR&#x27;25</td><td>â â</td><td>â â</td><td>â</td><td>â</td><td>â â</td><td>â</td><td></td><td></td><td></td><td></td><td>Â¸</td><td></td></tr><tr><td>ConcealGS [13]</td><td>ICASSP&#x27;25</td><td>â â</td><td>â â</td><td>â</td><td>â</td><td>â</td><td>â</td><td></td><td>â</td><td>â</td><td></td><td></td><td></td></tr><tr><td>Splats in Splats [61]</td><td>AAAI&#x27;26</td><td></td><td>Â¸â</td><td>â</td><td>â</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Â¸â</td><td></td></tr><tr><td>GS-Checker [24]</td><td>AAAI&#x27;26</td><td></td><td></td><td></td><td>3DGS Tampering Localization â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>â</td><td>3DGS Editing Safeguard</td><td></td><td>âÂ¸</td><td></td><td>â</td><td></td><td></td><td>â â</td><td>â</td></tr><tr><td>DEGauss [62]</td><td>NeurIPS&#x27;25</td><td></td><td></td><td></td><td>â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>â</td></tr><tr><td>AdLift [15]</td><td>arXiv&#x27;25</td><td></td><td></td><td></td><td>â</td><td></td><td></td><td></td><td>V</td><td></td><td>â â</td><td></td><td>â</td></tr></table>

(a) Attribute â Âµ: position; Î£: covariance; Î±: opacity; c: SH coefficients; a: auxiliary attributes. Distribution â Glob: perturbations embedded into all Gaussians; Loc:  
perturbations embedded into a subset of Gaussians; Freq: perturbations embed via fourier frequency domain; Unce: uncertainty-guided embedding. Inject â FT: asset-wise fine-tuning; Map: generalizable mapping; Gen: in-generation injection.  
(b) M-Dim (embedded message dimension) â 1D: bit sequence/text, 2D: image, 3D: 3D data; P-Dim (protective dimension): whether the task-specific protection can be conduct on 2D or 3D.  
(c) Aug: robustness augmentations applied in the 2D or 3D domain in training. 2D Attacks â JPEG: jpeg compression; Noise: Gaussian noise; Blur: Gaussian blur; Crop; Rot: rotation; Scale; Photo: photometric (brightness, contrast); Comb: combined distortions; GenP: generative purification; GenE: generative editing. 3D Attacks â Prune: sparsification, dropout, crop; Noise; Rot: rotation; Trans: translation; Scale; Comp: model compression/quantization; Clone: asset cloning; GenP: generative purification; GenE: generative editing.  
â indicates the method satisfies the corresponding criterion; blank indicates it does not; â indicates the criterion is not applicable to this method.  
Table 1: Summary of 24 existing 3DGS IP protection methods categorized by task type and characterized by: (a) Gaussian-based perturbation strategies, (b) task-specific protective capabilities, and (c) robustness.

Fourier frequency guidance. Recent methods observe that different frequency domains exhibit distinct robustness and fidelity trade-offs, assigning high frequency to visual fidelity and low frequency to robustness [63]. In general, highfrequency components are closely tied to fine-grained appearance details and are therefore more perceptually sensitive, while low-frequency components encode more stable structural and semantic information [56]. Perturbations embedded in high-frequency domains are often fragile, as they are suppressed by common compression pipelines [49, 55] and disrupted by reconstruction or generative editing processes [14, 64]. As a result, purely high-frequency embedding is typically avoided in robustness-oriented protection. In contrast, several methods prioritize low-frequency embedding to enhance survivability under distortions [14, 64]. To balance the complementary properties of different frequency domains, hybrid-frequency local strategies have also been explored [49, 55]. This hybrid design demonstrates that coordinating perturbations across multiple frequency domains can achieve a more favorable balance between imperceptibility and robustness than single-domain strategies.

Uncertainty guidance. In 3DGS, uncertainty reflects the sensitivity of rendering quality to perturbations in individual

Gaussian primitives. GaussianMarker [12] and MarkSplatter [50] exploit this property by embedding watermarks into Gaussians with high uncertainty, which contribute less to the final rendered appearance, such as those near object boundaries. Since perturbations applied to these Gaussians induce minimal perceptual changes, such strategies enable imperceptible modifications while maintaining reliable message extraction. However, because the number and spatial distribution of high-uncertainty Gaussians are inherently limited, such strategies often face constraints in achievable embedding capacity and flexibility.

Global strategy. The global strategy distributes perturbations uniformly across all Gaussian primitives without explicit subset selection. This design is widely adopted in existing methods [15, 24, 58, 62], particularly in steganographic settings [23, 58] where high embedding capacity and strong protection ability are required [15]. By leveraging the entire Gaussian representation, global strategies maximize information capacity and avoid dependence on specific spatial or spectral regions. However, because perturbations are applied broadly, this strategy typically requires careful optimization to preserve visual fidelity and prevent perceptible degradation [15, 62].

## 3.3 Perturbation Injection Pipeline

Existing 3DGS IP protection methods mainly adopt three pipelines for perturbation injection, including asset-wise finetuning, generalizable mapping, and in-generation injection.

Asset-wise fine-tuning. The basic but broadly adopted strategy for per-asset protection is asset-wise fine-tuning. This strategy is flexible because it can directly optimize Gaussian parameters with customizable objectives and distortion models, enabling scene-specific robustness-fidelity trade-offs [15,21,24]. However, the required per-asset optimization limits generalization to new assets [50] and incurs time costs that scale linearly with the number of assets.

Generalizable mapping. Generalizable feed-forward mapping trains a universal encoder on large-scale 3DGS datasets, thus enabling perturbation injection into arbitrary novel 3DGS assets in a single forward pass [50, 54, 57]. Several recent 3DGS watermarking methods adopt this strategy [50,57]. However, complex perturbation modeling introduces training instability [57], and joint encoder-decoder training on large-scale 3D datasets incurs substantial multi-GPU overhead [50, 57], leading to high initial deployment costs.

In-generation injection. In-generation injection integrates perturbations directly into 3DGS generative models [56, 60, 65], enabling assets to be born with protective perturbations at creation time. This pipeline achieves high efficiency for large-scale content production [60] and strong robustness, as perturbations are deeply embedded within the generative pipeline, either via cross-attention on intermediate features [60] or through U-Net block outputs [56]. However, such approaches can only protect assets generated by the corresponding 3DGS generative model and are not applicable to pre-existing content.

## 4 Passive 3DGS IP Protection

Passive 3DGS IP protection enables post-hoc verification and traceability without actively restricting 3DGS asset functionality. In this section, we summarize three passive protection paradigms: 3DGS watermarking, 3DGS steganography, and 3DGS tampering localization.

## 4.1 3DGS Watermarking

As 3DGS assets becomes increasingly circulated, reused, and redistributed across creative and commercial workflows, ensuring provable ownership and traceable usage of assets has become a critical requirement. 3DGS Watermarking fulfills this need by embedding verifiable identifiers that enable reliable copyright attribution even after assets undergo transformations or distribution through untrusted channels.

The task of 3DGS watermarking is to embed a binary message m $\in \mathcal { M } \subset \{ 0 , 1 \} ^ { L }$ sampled from a length-L space $\mathcal { M }$ into an original 3DGS asset $\mathbf { \bar { \boldsymbol { G } } }$ to obtain a protected (watermarked) asset $f _ { \theta } ( G )$ . As such, the task-specific scoring function $s ( \cdot ; \tau )$ in Equation (1) can be instantiated as the loglikelihood of correctly decoding m from the protected asset:

$$
s ( f _ { \theta } ( G ) ; \tau ) = \log P \big ( \mathbf { m } \big | D \big ( \Phi ( f _ { \theta } ( G ) ) \big ) \big ) ,\tag{2}
$$

where D denotes a watermark decoder and Î¦ is a 2D/3D feature interface that maps the (possibly rendered) protected asset $f _ { \theta } ( G )$ into the input space of D.

As shown in Table 1, existing 3DGS watermarking methods embed low-capacity binary identifiers into Gaussian representations to enable ownership verification under distortions, typically by coupling watermark objectives with 3DGS optimization. GaussianMarker [12] introduces uncertainty-aware embedding by modulating perturbations using gradient-based uncertainty, jointly perturbing positions and SH coefficients to resist noise and pruning. GuardSplat [21] improves robustness via adversarial training with simulated compression and noise, focusing on high-contribution Gaussians. RDSplat [14] targets diffusionbased generative editing by embedding watermarks into lowfrequency covariance components. 3D-GSW [49] leverages frequency-guided densification and adaptive attribute weighting, while NGS-Marker [52] embeds watermarks into local primitives to support partial infringement protection. Comp-MarkGS [55] emphasizes compression robustness through quantization-aware training on anchor features. To improve efficiency and generalization, GS-Marker [57] adopts a dualnetwork design enabling cross-scene watermarking, Mark-Splatter [50] reformulates Gaussians into grid-based splatter images for CNN-based embedding, and WATER-GS [48] supports both white-box and black-box extraction via viewagnostic optimization. Beyond direct embedding, Mantle-Mark [51] migrates watermarks from multi-view images during reconstruction, while GaussianSeal [56] integrates watermarking into generative 3DGS pipelines to produce bornsecure assets.

## 4.2 3DGS Steganography

Certain scenarios require covert and non-disruptive embedding of auxiliary information (e.g., licensing metadata, user identifiers, or scene annotations) for purposes such as secure communication or provenance binding. Steganography fulfills this need by enabling high-capacity and imperceptible embedding of multimodal payloads into 3DGS assets.

The task of 3DGS steganography is to embed a secret payload $\mathbf { p } \in \mathcal { P }$ sampled from a payload space ${ \mathcal { P } } \left( \mathrm { e . g . } \right.$ , a bit sequence, a 2D image, or a 3D Gaussian/volumetric asset) into a cover 3DGS asset G to obtain a stego counterpart $f _ { \theta } ( G , \mathbf { p } )$ We instantiate the task-specific scoring function s(Â·; Ï ) in Equation (1) as the negative payload reconstruction loss:

$$
\begin{array} { r } { s ( f _ { \theta } ( G , \mathbf { p } ) ; \tau ) = - \mathcal { L } _ { \mathrm { m s g } } \big ( \mathbf { p } , \mathcal { D } ( \Phi ( f _ { \theta } ( G , \mathbf { p } ) ) ) \big ) , } \end{array}\tag{3}
$$

where D denotes a stego payload decoder and Î¦ is a 2D/3D feature interface that maps the (possibly rendered) stego asset $f _ { \theta } ( G , \mathbf { p } )$ into the input space of D.

3DGS steganography aims to maximize payload capacity while preserving high visual fidelity, as summarized in Table 1. Unlike watermarking, steganographic methods prioritize imperceptibility and information throughput, with limited emphasis on robustness against deliberate distortions. Most approaches embed messages by directly modulating appearance-related Gaussian attributes. GS-Hider [22] and ConcealGS [13] achieve high-capacity embedding by modulating SH coefficients or color attributes with minimal visual degradation. Splats in Splats [61] further increases capacity by embedding 3D content via bit-shifting and opacity mapping, preserving structural integrity. GaussianStego [66] introduces a generalizable encoder-decoder pipeline for crossscene steganography. Hide A Bit [59] enables ultra-fast embedding through LSB manipulation combined with RSA encryption, without optimization. KeySS [58] enforces access control using a key-secured decoder and a 3D-Sinkhorn metric, while SecureGS [42] adopts a decoupled encryption scheme within Scaffold-GS anchors to balance security and fidelity. StegaGaussian [23] hides messages in highfrequency domains via frequency decomposition, whereas $\mathrm { { X - S G ^ { 2 } } }$ [54] exploits cross-modal redundancy. For dynamic scenes, Hide-in-Motion [67] embeds payloads into spatiotemporal deformation fields of 4DGS. Finally, InstantSplamp [60] performs in-generation steganography by injecting messages through cross-attention layers during generative inference.

## 4.3 3DGS Tampering Localization

As 3DGS assets undergo editing, compositing, and transformation across production pipelines, unauthorized or malicious modifications may compromise authenticity and create trustworthiness concerns. 3DGS tampering localization focuses on detecting and spatially identifying altered regions to support integrity verification and forensic analysis.

GS-Checker [24] is the first and currently the only 3DGS tampering localization method. It does not require preembedded perturbations; instead, it directly analyzes tampered 3DGS assets. By attaching a 3D tampering attribute to each Gaussian and employing a 3D contrastive mechanism to reveal local inconsistencies, GS-Checker detects suspect manipulations directly in the 3D domain.

## 5 Active 3DGS IP Protection

Active 3DGS IP protection proactively limits unauthorized downstream use of 3DGS assets [26]. Current active methods predominantly target editing safeguards.

## 5.1 3DGS Editing Safeguard

Recent 3DGS editing works enable faithful manipulation of 3DGS assets and advance 3D content creation [68â72]. However, they also expose 3DGS assets to significant risks of unauthorized or malicious editing, potentially leading to identity deception, misinformation, or reputational damage [47, 73, 74]. This highlights the urgency of developing effective 3DGS editing safeguard methods.

The task of 3DGS editing safeguard to learn a safeguard operator $f _ { \theta }$ that produces a protected asset $f _ { \theta } ( G )$ for an original 3DGS asset G such that instruction-driven editing either induces minimal semantic change or drastically degrades editing quality [75]. Thus, the task-specific scoring function $s ( \cdot ; \tau )$ in Equation (1) can be written as the negative expected editing success under an editing model $\mathcal { E } _ { \phi } \mathrm { : }$

$$
\begin{array} { r } { s ( f _ { \theta } ( G ) ; \tau ) = - \mathbb { E } _ { \mathbf { e } } \left[ \mathcal { L } _ { \mathrm { e d i t } } ( \mathcal { E } _ { \phi } ( \Phi ( f _ { \theta } ( G ) ) , \mathbf { e } ) , \Phi ( G ) ) \right] , } \end{array}\tag{4}
$$

where e is an editing instruction sampled from an instruction space E, Î¦ is a 2D/3D feature interface into the editing model, and $\mathcal { L } _ { \mathrm { e d i t } }$ measures editing success (e.g., semantic deviation or instruction alignment).

Equation (4) can be seen as an adversarial attack for the editing model [47, 73, 75]. AdLift [15] prevents instructiondriven editing by lifting strictly bounded 2D adversarial perturbations into a 3D Gaussian safeguard. It optimizes these safeguard Gaussians via a tailored lifted projected gradient descent (PGD) [76â80] that truncates gradients from the editing model at the rendered image level and applies projected gradients to enforce strictly image-space invisable constraints. The final perturbations are propagated to Gaussians through an image-to-Gaussian fitting stage. DEGauss [62] improves cross-view robustness through a view-focal gradient fusion module that prioritizes gradients from challenging viewpoints. It further enhances the adversarial objective via dual discrepancy optimization, jointly maximizing semantic deviation and directional bias of the guidance signal to better disrupt editing trajectories.

## 6 Robustness Issues

Robustness is essential for 3DGS IP protection, as adversaries may attempt to weaken or erase protection through conventional distortions or generative-based methods, affecting either 2D renders or the 3DGS space. As shown in Figure 1(c), we summarize three robustness challenges considered in existing 3DGS IP protection: (i) conventional 2D and 3D distortions, (ii) generative purification, and (iii) generative editing.

Conventional 2D and 3D distortions. Since perturbations embedded in 3DGS are ultimately projected onto 2D rendered images, conventional 2D distortions [81] can effectively disrupt such perturbations. These include geometric transformations (e.g., rotation and cropping [49]), photometric variations (e.g., brightness and contrast adjustment [50]), and signal degradations such as Gaussian blur, noise injection, and JPEG compression [12]. Moreover, 3D distortions include geometric distortions and degradation distortions. Geometric distortions explicitly alter the spatial configuration of Gaussian primitives. Pruning, cloning, and densification modify Gaussian density and distribution [22, 61], diluting protective perturbations. Spatial transformations including rotation, scaling, and translation perturb Gaussian attributes like positions and covariances, thereby misaligning geometry-dependent encodings [12, 52]. In contrast, degradation distortions impair fidelity: Gaussian noise corrupts geometry and appearance jointly [49,50], while model compression suppresses low-amplitude perturbations through quantization and parameter merging [42].

Generative purification. Generative purification methods [82, 83] aim to map inputs back to their natural data distribution, thereby intentionally eliminating protective perturbations while preserving fidelity. In the 2D domain, generative purification methods such as DiffPure [84], Impress [82], GrIDPure [83], and PDM-Pure [85] leverage diffusionbased reverse processes to systematically eliminate adversarial noise. Besides, 3D-specific methods focus on intrinsic 3DGS structures. Approaches like GSPure [31] and GMEA [86] exploit the redundancy of Gaussians, pruning elements that are discriminative for decoding yet contribute minimally to reconstruction. These methods effectively compromise mainstream schemes [12, 22, 42, 61] by balancing visual fidelity with attack efficacy, highlighting generative purification as a critical yet underexplored vulnerability.

Generative editing. The emgerging generative editing methods poses a severe challenge to 3DGS IP protection. Diffusion-based 2D editing [87â89] and 3D editing [68â72] operate as powerful semantic modifiers. These editing models typically function as aggressive low-pass filters [14, 64], filtering out high-frequency protection perturbations in 3DGS assets [14]. Most recent 3DGS watermarking method [14] exploits this observation and explicitly embed watermarks into low-frequency covariance components, ensuring robust 3DGS watermarking against generative editing. Alternatively, 3DGS editing safeguard approaches [15, 62] adopt an active defense strategy to disrupt the unauthorized editing process itself.

## 7 Future Directions and Challenges

Although 3DGS IP protection has made notable progress, numerous unaddressed challenges remain. In this section, we outline six key future directions with their challenges, grouped into three thematic areas: robustness (R1-3), efficiency (E1-2), and protection paradigms (P1).

R1: Resistance to generative purification and editing. Generative 2D purification [82â85] and emerging 3D purification strategies [31] can effectively eliminate invisible protective signals, whereas most existing methods do not explicitly incorporate purification-aware training or evaluation, resulting in fragile real-world robustness. Besides, for generative editing [68â72], current defense [14, 15] often compromise visual quality to achieve resistance against diffusionbased editing [14]. Pushing the frontier to achieve both high imperceptibility and resilience against generative distortions remains a grand challenge.

R2: The cross-representation survival problem. In realworld distribution pipelines, 3DGS assets are frequently converted into other formats such as meshes, voxels, or NeRF [4]. Most existing methods fail to survive these representation conversions. While initial attempts like MantleMark [51] explore 2D-to-3D survival, future research must explore representation-agnostic perturbations techniques [90, 91]. These should embed information into high-level geometric topology or frequency domains [23], ensuring the watermark persists even when the 3DGS is converted to other formats [4, 20, 67].

R3: Unified benchmarking. The lack of a standardized evaluation framework hinders fair comparison across methods. Existing works rely on different datasets, attack configurations, and evaluation metrics. As shown in Table 1, most methods do not consider or evaluate robustness against diverse attack types. To enable systematic progress, the community urgently needs a unified benchmark that provides a standardized attack suite (ranging from conventional distortions to generative attacks), a fixed set of diverse test scenes, and unified metrics for imperceptibility and robustness.

E1: Generalizable and universal perturbations. Most 3DGS IP protection methods [21, 49] rely on per-asset finetuning, which is computationally expensive and impractical for large-scale asset pipelines. An important future direction is the development of generalizable mappings or universal perturbations [50,57]. Such approaches should be pre-trained on large-scale 3D datasets, enabling perturbations to be injected into arbitrary 3DGS assets in a single forward pass at inference time, without the need for per-asset fine-tuning.

E2: Built-in security for reconstruction and generative 3DGS. As the primary source of 3D assets shifts to feedforward reconstruction [92] and generative pipelines [65], IP protection must shift accordingly. Future feed-forward reconstruction and large-scale 3DGS generative models should incorporate built-in IP protective capabilities [56, 60], where perturbation mechanisms are integrated directly into the modelâs weights or latent space. Consequently, such models would produce 3DGS assets that are inherently protected, potentially eliminating the need for post-hoc protection.

P1: Protection paradigm. Existing research on 3DGS IP protection mainly focus on post-hoc verification paradigms such as 3DGS watermarking [12] and 3DGS steganography [66], which focus on ownership identification rather than usage control. In contrast, active protection, i.e., restricting a protected assetâs usability for downstream tasks, remains largely underexplored, with recent progress only beginning to address anti-editing protection against instruction-driven manipulation [15, 62]. However, comprehensive lifecycle asset protection should span the entire pipeline, from initial non-usability to post-training deletability [26], encompassing mechanisms such as unlearnability [43], non-transferability [44, 93â98], and data forgetting [99â102], among others. Developing such lifecycle-aware protection mechanisms is foundational for establishing a healthy ecosystem for 3DGS asset IP governance in the AIGC era, enabling technical enforceability, traceability, and legal auditability.

## 8 Conclusion

In this survey, we provide the first structured and comprehensive overview of 3DGS IP protection. We introduce a bottomup framework that spans three complementary layers, from underlying Gaussian-based perturbation mechanisms, to passive and active 3DGS IP protection paradigms, and to emerging robustness threats in the AIGC era. We also outline key research challenges and future opportunities across robustness, efficiency, and protection paradigms, with the goal of facilitating trustworthy and scalable deployment of 3DGS assets. We hope this survey serves as a foundational step toward building a trustworthy and secure ecosystem for 3DGS IP applications and enables future research to develop more robust, efficient, and deployable protection solutions.

## References

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[2] Xiangyu Sun, Runnan Chen, Mingming Gong, Dong Xu, and Tongliang Liu. Intern-gs: Vision model guided sparseview 3d gaussian splatting. arXiv preprint arXiv:2505.20729, 2025.

[3] Haodong Chen, Runnan Chen, Qiang Qu, Zhaoqing Wang, Tongliang Liu, Xiaoming Chen, and Yuk Ying Chung. Beyond gaussians: Fast and high-fidelity 3d splatting with linear kernels. arXiv preprint arXiv:2411.12440, 2024.

[4] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[5] Ben Fei, Jingyi Xu, Rui Zhang, Qingyuan Zhou, Weidong Yang, and Ying He. 3d gaussian splatting as new era: A survey. IEEE Transactions on Visualization and Computer Graphics, 2024.

[6] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstrom, Ste- Â¨ fano Mattoccia, Martin R Oswald, and Matteo Poggi. How nerfs and 3d gaussian splatting are reshaping slam: a survey. arXiv preprint arXiv:2402.13255, 4:1, 2024.

[7] Runnan Chen, Xiangyu Sun, Zhaoqing Wang, Youquan Liu, Jiepeng Wang, Lingdong Kong, Jiankang Deng, Mingming Gong, Liang Pan, Wenping Wang, et al. Ovgaussian: Generalizable 3d gaussian segmentation with open vocabularies. arXiv preprint arXiv:2501.00326, 2024.

[8] Jiaxin Huang, Ziwen Li, Hanlve Zhang, Runnan Chen, Xiao He, Yandong Guo, Wenping Wang, Tongliang Liu, and Mingming Gong. Surprise3d: A dataset for spatial understanding and reasoning in complex 3d scenes. arXiv preprint arXiv:2507.07781, 2025.

[9] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1â18. Springer, 2024.

[10] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10324â10335, 2024.

[11] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li, Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting: Survey, technologies, challenges, and opportunities. IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[12] Xiufeng Huang, Ruiqi Li, Yiu-ming Cheung, Ka Chun Cheung, Simon See, and Renjie Wan. Gaussianmarker: Uncertainty-aware copyright protection of 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:33037â33060, 2024.

[13] Yifeng Yang, Hengyu Liu, Chenxin Li, Yining Sun, Wuyang Li, Yifan Liu, Yiyang Lin, Yixuan Yuan, and Nanyang Ye. Concealgs: Concealing invisible copyright information in 3d gaussian splatting. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1â5. IEEE, 2025.

[14] Longjie Zhao, Ziming Hong, Zhenyang Ren, Runnan Chen, Mingming Gong, and Tongliang Liu. Rdsplat: Robust watermarking against diffusion editing for 3d gaussian splatting. arXiv preprint arXiv:2512.06774, 2025.

[15] Ziming Hong, Tianyu Huang, Runnan Chen, Shanshan Ye, Mingming Gong, Bo Han, and Tongliang Liu. Adlift: Lifting adversarial perturbations to safeguard 3d gaussian splatting assets against instruction-driven editing. arXiv preprint arXiv:2512.07247, 2025.

[16] Ziyuan Luo, Qing Guo, Ka Chun Cheung, Simon See, and Renjie Wan. Copyrnerf: Protecting the copyright of neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22401â22411, 2023.

[17] Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, and Renjie Wan. Protecting nerfsâ copyright via plug-and-play watermarking base model. In European Conference on Computer Vision, pages 57â73. Springer, 2024.

[18] Ziyuan Luo, Anderson Rocha, Boxin Shi, Qing Guo, Haoliang Li, and Renjie Wan. The nerf signature: Codebookaided watermarking for neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[19] Xiufeng Huang, Ka Chun Cheung, Simon See, and Renjie Wan. Geometrysticker: Enabling ownership claim of recolorized neural radiance fields. In European Conference on Computer Vision, pages 438â454. Springer, 2024.

[20] Xianlong Wang, Minghui Li, Wei Liu, Hangtao Zhang, Shengshan Hu, Yechao Zhang, Ziqi Zhou, and Hai Jin. Unlearnable 3d point clouds: Class-wise transformation is all you need. Advances in Neural Information Processing Systems, 37:99404â99432, 2024.

[21] Zixuan Chen, Guangcong Wang, Jiahao Zhu, Jianhuang Lai, and Xiaohua Xie. Guardsplat: Efficient and robust watermarking for 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16325â16335, 2025.

[22] Xuanyu Zhang, Jiarui Meng, Runyi Li, Zhipei Xu, Yongbing Zhang, and Jian Zhang. Gs-hider: Hiding messages into 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:49780â49805, 2024.

[23] Feng Wang, Wenjing Feng, Jiayan Wang, Yong Tang, and Jing Zhao. Stegagaussian: High-fidelity steganography for 3d gaussian splatting based on frequency decomposition. Knowledge-Based Systems, page 114863, 2025.

[24] Haoliang Han, Ziyuan Luo, Jun Qi, Anderson Rocha, and Renjie Wan. Gs-checker: Tampering localization for 3d gaussian splatting. arXiv preprint arXiv:2511.20354, 2025.

[25] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[26] Yiming Li, Shuo Shao, Yu He, Junfeng Guo, Tianwei Zhang, Zhan Qin, Pin-Yu Chen, Michael Backes, Philip Torr, Dacheng Tao, et al. Rethinking data protection in the (generative) artificial intelligence era. arXiv preprint arXiv:2507.03034, 2025.

[27] Shiming Chen, Ziming Hong, Xinge You, and Ling Shao. Semantics-conditioned generative zero-shot learning via feature refinement. International Journal of Computer Vision, pages 1â18, 2025.

[28] Yexiong Lin, Yu Yao, and Tongliang Liu. Beyond optimal transport: Model-aligned coupling for flow matching. arXiv preprint arXiv:2505.23346, 2025.

[29] Dingjie Fu, Wenjin Hou, Shiming Chen, Shuhuang Chen, Xinge You, Salman Khan, and Fahad Shahbaz Khan. Discriminative image generation with diffusion models for zeroshot learning. arXiv preprint arXiv:2412.17219, 2024.

[30] Wenjin Hou, Shiming Chen, Shuhuang Chen, Ziming Hong, Yan Wang, Xuetao Feng, Salman Khan, Fahad Shahbaz Khan, and Xinge You. Visual-augmented dynamic semantic

prototype for generative zero-shot learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23627â23637, 2024.

[31] Wenkai Huang, Yijia Guo, Gaolei Li, Lei Ma, Hang Zhang, Liwen Hu, Jiazheng Wang, Jianhua Li, and Tiejun Huang. Can protective watermarking safeguard the copyright of 3d gaussian splatting? arXiv preprint arXiv:2511.22262, 2025.

[32] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19740â 19750, 2023.

[33] Runnan Chen, Zhaoqing Wang, Jiepeng Wang, Yuexin Ma, Mingming Gong, Wenping Wang, and Tongliang Liu. Panoslam: Panoptic 3d scene reconstruction via gaussian slam. arXiv preprint arXiv:2501.00352, 2024.

[34] Ziwen Li, Jiaxin Huang, Runnan Chen, Yunlong Che, Yandong Guo, Tongliang Liu, Fakhri Karray, and Mingming Gong. Urbangs: Semantic-guided gaussian splatting for urban scene reconstruction. arXiv preprint arXiv:2412.03473, 2024.

[35] Jiaxin Huang, Runnan Chen, Ziwen Li, Zhengqing Gao, Xiao He, Yandong Guo, Mingming Gong, and Tongliang Liu. Mllm-for3d: Adapting multimodal large language model for 3d reasoning segmentation. arXiv, 2025.

[36] Tianyu Huang, Runnan Chen, Dongting Hu, Fengming Huang, Mingming Gong, and Tongliang Liu. Openinsgaussian: Open-vocabulary instance gaussian segmentation with context-aware cross-view fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6341â6350, 2025.

[37] Andrew B Kahng, John Lach, William H Mangione-Smith, Stefanus Mantik, Igor L Markov, Miodrag Potkonjak, Paul Tucker, Huijuan Wang, and Gregory Wolfe. Watermarking techniques for intellectual property protection. In Proceedings of the 35th annual Design Automation Conference, pages 776â781, 1998.

[38] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In Proceedings of the European conference on computer vision (ECCV), pages 657â672, 2018.

[39] Yinlong Qian, Jing Dong, Wei Wang, and Tieniu Tan. Deep learning for steganalysis via convolutional neural networks. In Media Watermarking, Security, and Forensics 2015, volume 9409, pages 171â180. SPIE, 2015.

[40] Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan. Mantra-net: Manipulation tracing network for detection and localization of image forgeries with anomalous features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9543â9552, 2019.

[41] Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, and Bo Li. Domain watermark: Effective and harmless dataset copyright protection is closed at hand. Advances in Neural Information Processing Systems, 36:54421â54450, 2023.

[42] Xuanyu Zhang, Jiarui Meng, Zhipei Xu, Shuzhou Yang, Yanmin Wu, Ronggang Wang, and Jian Zhang. Securegs: Boosting the security and fidelity of 3d gaussian splatting steganography. arXiv preprint arXiv:2503.06118, 2025.

[43] Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, and Yisen Wang. Unlearnable examples: Making personal data unexploitable. arXiv preprint arXiv:2101.04898, 2021.

[44] Ziming Hong, Yongli Xiang, and Tongliang Liu. Toward robust non-transferable learning: A survey and benchmark. IJ-CAI, 2025.

[45] Zihan Wang, Zhiyong Ma, Zhongkui Ma, Shuofeng Liu, Akide Liu, Derui Wang, Minhui Xue, and Guangdong Bai. Catch-only-one: Non-transferable examples for modelspecific authorization. arXiv preprint arXiv:2510.10982, 2025.

[46] Xinfeng Li, Tianze Qiu, Yingbin Jin, Lixu Wang, Hanqing Guo, Xiaojun Jia, Xiaofeng Wang, and Wei Dong. Webcloak: Characterizing and mitigating threats from llm-driven web agents as intelligent scrapers. In Proceedings of the 2026 IEEE Symposium on Security and Privacy (SP), 2026.

[47] Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples. arXiv preprint arXiv:2302.04578, 2023.

[48] Yuqi Tan, Xiang Liu, Shuzhao Xie, Bin Chen, Shu-Tao Xia, and Zhi Wang. Water-gs: Toward copyright protection for 3d gaussian splatting via universal watermarking. arXiv preprint arXiv:2412.05695, 2024.

[49] Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, and Sangpil Kim. 3d-gsw: 3d gaussian splatting for robust watermarking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5938â5948, 2025.

[50] Xiufeng Huang, Ziyuan Luo, Qi Song, Ruofei Wang, and Renjie Wan. Marksplatter: Generalizable watermarking for 3d gaussian splatting model via splatter image structure. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12189â12198, 2025.

[51] Ziyuan Luo, Jun Liu, Haoliang Li, Anderson Rocha, and Renjie Wan. Mantlemark: Migrating watermarks from multiview images to radiance fields via frequency modulation. IEEE Transactions on Information Forensics and Security, 2025.

[52] Anonymous. NGS-marker: Robust native watermarking for 3d gaussian splatting. In The Fourteenth International Conference on Learning Representations, 2026.

[53] Rui Xu, Gaolei Li, Wenkai Huang, Jiazheng Wang, Hang Zhang, and Jianhua Li. Mark3dgs: Protecting the intellectual property of 3d gaussian splatting with robust watermarking, 2025.

[54] Zihang Cheng, Huiping Zhuang, Chun Li, Xin Meng, Ming Li, Fei Richard Yu, and Liqiang Nie. X-sg2s: Safe and generalizable gaussian splatting with x-dimensional watermarks. arXiv preprint arXiv:2502.10475, 2025.

[55] Sumin In, Youngdong Jang, Utae Jeong, MinHyuk Jang, Hyeongcheol Park, Eunbyung Park, and Sangpil Kim. Compmarkgs: Robust watermarking for compressed 3d gaussian splatting. arXiv preprint arXiv:2503.12836, 2025.

[56] Runyi Li, Xuanyu Zhang, Chuhan Tong, Zhipei Xu, and Jian Zhang. Gaussianseal: Rooting adaptive watermarks for 3d gaussian generation model. arXiv preprint arXiv:2503.00531, 2025.

[57] Lijiang Li, Jinglu Wang, Xiang Ming, and Yan Lu. Gsmarker: Generalizable and robust watermarking for 3d gaussian splatting. arXiv preprint arXiv:2503.18718, 2025.

[58] Yan Ren, Shilin Lu, and Adams Wai-Kin Kong. All that glitters is not gold: Key-secured 3d secrets within 3d gaussian splatting. arXiv preprint arXiv:2503.07191, 2025.

[59] Kaoru Sasaki, Kazuhito Sato, Shugo Yamaguchi, Keitaro Tanaka, and Shigeo Morishima. Hide a bit: A training-free and high-fidelity steganography method for 3d gaussian splatting based on bit manipulation and rsa encryption. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Posters, pages 1â2, 2025.

[60] Chenxin Li, Hengyu Liu, Zhiwen Fan, Wuyang Li, Yifan Liu, Panwang Pan, and Yixuan Yuan. Instantsplamp: Fast and generalizable stenography framework for generative gaussian splatting. In The Thirteenth International Conference on Learning Representations, 2025.

[61] Yijia Guo, Wenkai Huang, Yang Li, Gaolei Li, Hang Zhang, Liwen Hu, Jianhua Li, Tiejun Huang, and Lei Ma. Splats in splats: Embedding invisible 3d watermark within gaussian splatting. arXiv preprint arXiv:2412.03121, 2024.

[62] Lingzhuang Meng, Mingwen Shao, Yuanjian Qiao, and Xiang Lv. DEGauss: Defending against malicious 3d editing for gaussian splatting. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[63] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. Gaussianeditor: Editing 3d gaussians delicately with text instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20902â 20911, 2024.

[64] Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong. Robust watermarking using generative priors against image editing: From benchmarking to advances. arXiv preprint arXiv:2410.18775, 2024.

[65] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21469â21480, 2025.

[66] Chenxin Li, Hengyu Liu, Zhiwen Fan, Wuyang Li, Yifan Liu, Panwang Pan, and Yixuan Yuan. Gaussianstego: A generalizable stenography pipeline for generative 3d gaussians splatting. arXiv preprint arXiv:2407.01301, 2024.

[67] Hengyu Liu, Chenxin Li, Wentao Pan, Zhiqin Yang, Yifeng Yang, Yifan Liu, Wuyang Li, and Yixuan Yuan. Hide-inmotion: Embedding steganographic copyright information into 4d gaussian splatting assets. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 2694â2700. IEEE, 2025.

[68] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing. In European Conference on Computer Vision, pages 55â 71. Springer, 2024.

[69] Dong In Lee, Hyeongcheol Park, Jiyoung Seo, Eunbyung Park, Hyunje Park, Ha Dam Baek, Sangheon Shin, Sangmin Kim, and Sangpil Kim. Editsplat: Multi-view fusion and attention-guided optimization for view-consistent 3d scene editing with 3d gaussian splatting. In Proceedings of the

Computer Vision and Pattern Recognition Conference, pages 11135â11145, 2025.

[70] Minghao Chen, Iro Laina, and Andrea Vedaldi. Dge: Direct gaussian 3d editing by consistent multi-view editing. In European Conference on Computer Vision, pages 74â92. Springer, 2024.

[71] Yuxuan Wang, Xuanyu Yi, Zike Wu, Na Zhao, Long Chen, and Hanwang Zhang. View-consistent 3d editing with gaussian splatting. In European conference on computer vision, pages 404â420. Springer, 2024.

[72] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21476â21485, 2024.

[73] Haotian Xue, Chumeng Liang, Xiaoyu Wu, and Yongxin Chen. Toward effective protection against diffusion-based mimicry through score distillation. In The Twelfth International Conference on Learning Representations, 2024.

[74] Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y Zhao. Glaze: Protecting artists from style mimicry by {Text-to-Image} models. In 32nd USENIX Security Symposium (USENIX Security 23), pages 2187â2204, 2023.

[75] Chumeng Liang and Xiaoyu Wu. Mist: Towards improved adversarial examples for diffusion models. arXiv preprint arXiv:2305.12683, 2023.

[76] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.

[77] Zhuo Huang, Xiaobo Xia, Li Shen, Bo Han, Mingming Gong, Chen Gong, and Tongliang Liu. Harnessing outof-distribution examples via augmenting content and style. arXiv preprint arXiv:2207.03162, 2022.

[78] Runqi Lin, Chaojian Yu, and Tongliang Liu. Eliminating catastrophic overfitting via abnormal adversarial examples regularization. Advances in Neural Information Processing Systems, 36:67866â67885, 2023.

[79] Runqi Lin, Chaojian Yu, Bo Han, and Tongliang Liu. On the over-memorization during natural, robust and catastrophic overfitting. arXiv preprint arXiv:2310.08847, 2023.

[80] Runqi Lin, Chaojian Yu, Bo Han, Hang Su, and Tongliang Liu. Layer-aware analysis of catastrophic overfitting: Revealing the pseudo-robust shortcut dependency. arXiv preprint arXiv:2405.16262, 2024.

[81] Zhuo Huang, Miaoxi Zhu, Xiaobo Xia, Li Shen, Jun Yu, Chen Gong, Bo Han, Bo Du, and Tongliang Liu. Robust generalization against photon-limited corruptions via worst-case sharpness minimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16175â16185, 2023.

[82] Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, and Jinghui Chen. Impress: Evaluating the resilience of imperceptible perturbations against unauthorized data usage in diffusion-based generative ai. Advances in Neural Information Processing Systems, 36:10657â10677, 2023.

[83] Zhengyue Zhao, Jinhao Duan, Kaidi Xu, Chenan Wang, Rui Zhang, Zidong Du, Qi Guo, and Xing Hu. Can protective perturbation safeguard personal data from being exploited by stable diffusion? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24398â 24407, 2024.

[84] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. arXiv preprint arXiv:2205.07460, 2022.

[85] Haotian Xue and Yongxin Chen. Pixel is a barrier: Diffusion models are more adversarially robust than we think. arXiv preprint arXiv:2404.13320, 2024.

[86] Qingyuan Zeng, Shu Jiang, Jiajing Lin, Zhenzhong Wang, Kay Chen Tan, and Min Jiang. Fading the digital ink: A universal black-box attack framework for 3dgs watermarking systems. arXiv preprint arXiv:2508.07263, 2025.

[87] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392â18402, 2023.

[88] Zhenchen Wan, Yanwu Xu, Dongting Hu, Weilun Cheng, Tianxi Chen, Zhaoqing Wang, Feng Liu, Tongliang Liu, and Mingming Gong. Mft-viton: High-fidelity virtual try-on with minimal input via a mask-free transformer-diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1985â1994, 2025.

[89] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836â3847, 2023.

[90] Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, and Renjie Wan. Geometry cloak: Preventing tgs-based 3d reconstruction from copyrighted images. Advances in Neural Information Processing Systems, 37:119361â119385, 2024.

[91] Zhuo Huang, Gang Niu, Bo Han, Masashi Sugiyama, and Tongliang Liu. Towards out-of-modal generalization without instance-level modal correspondence. In The Thirteenth International Conference on Learning Representations, 2025.

[92] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG), 44(6):1â16, 2025.

[93] Lixu Wang, Shichao Xu, Ruiqi Xu, Xiao Wang, and Qi Zhu. Non-transferable learning: A new approach for model ownership verification and applicability authorization. arXiv preprint arXiv:2106.06916, 2021.

[94] Ziming Hong, Runnan Chen, Zengmao Wang, Bo Han, Bo Du, and Tongliang Liu. When data-free knowledge distillation meets non-transferable teacher: Escaping out-of-distribution trap is all you need. arXiv preprint arXiv:2507.04119, 2025.

[95] Yongli Xiang, Ziming Hong, Lina Yao, Dadong Wang, and Tongliang Liu. Jailbreaking the non-transferable barrier via test-time data disguising. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 30671â 30681, 2025.

[96] Ziming Hong, Li Shen, and Tongliang Liu. Your transferability barrier is fragile: Free-lunch for transferring the nontransferable learning. In Proceedings of the IEEE/CVF Con-

ference on Computer Vision and Pattern Recognition, pages 28805â28815, 2024.

[97] Ziming Hong, Zhenyi Wang, Li Shen, Yu Yao, Zhuo Huang, Shiming Chen, Chuanwu Yang, Mingming Gong, and Tongliang Liu. Improving non-transferable representation learning by harnessing content and style. In The twelfth international conference on learning representations, 2024.

[98] Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, and Wei Dong. Token buncher: Shielding llms from harmful reinforcement learning fine-tuning. arXiv preprint arXiv:2508.20697, 2025.

[99] Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In 2021 IEEE symposium on security and privacy (SP), pages 141â159. IEEE, 2021.

[100] Shixuan Wang, Jingwen Ye, and Xinchao Wang. Machine unlearning in 3d generation: A perspective-coherent acceleration framework. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[101] Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, and Bo Han. Exploring criteria of loss reweighting to enhance llm unlearning. arXiv preprint arXiv:2505.11953, 2025.

[102] Chongyang Gao, Lixu Wang, Kaize Ding, Chenkai Weng, Xiao Wang, and Qi Zhu. On large language model continual unlearning. arXiv preprint arXiv:2407.10223, 2024.