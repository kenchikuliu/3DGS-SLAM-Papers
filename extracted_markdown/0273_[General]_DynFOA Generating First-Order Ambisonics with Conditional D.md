<!-- page 1 -->
DynFOA: Generating First-Order Ambisonics with Conditional Diffusion
for Dynamic and Acoustically Complex 360-Degree Videos
Ziyu Luo1
Lin Chen1
Qiang Qu2
Xiaoming Chen1
Yiran Shen3
1School of Computer and Artificial Intelligence, Beijing Technology and Business University, Beijing, China
2School of Computer Science, The University of Sydney, Sydney, NSW, Australia
3School of Software, Shandong University, Jinan, China
vincent.qu@sydney.edu.au, xiaoming.chen@btbu.edu.cn, yiran.shen@sdu.edu.cn
Figure 1: DynFOA Framework for Immersive First-Order Ambisonics (FOA) Generation. This pipeline processes 360-degree
video input through a multi-stage architecture: video preprocessing and segmentation, multi-modal feature extraction using
Convolutional Neural Networks (CNN) for visual features and Spherical Harmonic Transform (SHT) for spatial audio decom-
position, cross-modal feature fusion and alignment, geometric and acoustic scene analysis for material property estimation, and
spatially-aware sound source separation. The conditional diffusion generator then integrates these extracted features—including
geometry, material properties, and localized sound sources—with physics-based acoustic modeling to generate high-fidelity
FOA audio featuring realistic occlusion, reflections, and reverberation effects, delivering an immersive spatial audio experience.
ABSTRACT
Spatial audio is crucial for creating compelling immersive 360-
degree video experiences. However, generating realistic spatial au-
dio, such as first-order ambisonics (FOA), from 360-degree videos
in complex acoustic scenes remains challenging. Existing methods
often overlook the dynamic nature and acoustic complexity of 360-
degree scenes, fail to fully account for dynamic sound sources, and
neglect complex environmental effects such as occlusion, reflec-
tions, and reverberation, which are influenced by scene geometries
and materials. We propose DynFOA, a framework based on dy-
namic acoustic perception and conditional diffusion, for generating
high-fidelity FOA from 360-degree videos. DynFOA first performs
visual processing via a video encoder, which detects and localizes
multiple dynamic sound sources, estimates their depth and seman-
tics, and reconstructs the scene geometry and materials using a 3D
Gaussian Splatting. This reconstruction technique accurately mod-
els occlusion, reflections, and reverberation based on the geome-
tries and materials of the reconstructed 3D scene and the listener’s
viewpoint. The audio encoder then captures the spatial motion and
temporal 4D sound source trajectories to fine-tune the diffusion-
based FOA generator. The fine-tuned FOA generator adjusts spatial
cues in real time, ensuring consistent directional fidelity during lis-
tener head rotation and complex environmental changes. Extensive
evaluations demonstrate that DynFOA consistently outperforms ex-
isting methods across metrics such as spatial accuracy, acoustic fi-
delity, and distribution matching, while also improving the user ex-
perience. Therefore, DynFOA provides a robust and scalable ap-
proach to rendering realistic dynamic spatial audio for VR and im-
mersive media applications.
Index Terms:
360-Degree Video, First-Order Ambisonics, Dyn-
FOA, Conditional Diffusion.
1
INTRODUCTION
Spatial audio is a cornerstone of immersion in Virtual Reality (VR)
and 360-degree media, allowing users to perceive sound with re-
alistic direction, distance, and environmental context [1]. Unlike
conventional stereo audio, spatial audio aligns the auditory environ-
ment with visual cues, enabling a coherent perceptual scene. How-
ever, generating high-fidelity, physically plausible spatial audio re-
mains challenging. Realistic spatial audio requires not only precise
1
arXiv:2602.06846v2  [cs.SD]  27 Feb 2026

<!-- page 2 -->
synchronization with visual events but also accurate 3D localiza-
tion of sound sources, realistic modeling of occlusion, reflections,
and reverberation, and adaptation to the listener’s head orientation
in real time [2].
A major limitation in existing works is the under-utilization
of the rich geometric and semantic information available in 360-
degree video for audio rendering. Many methods, such as some
FOA-based approaches or hybrid audio-visual models, focus pri-
marily on sound source localization without modeling how the sur-
rounding environment shapes the sound field. For instance, Om-
niAudio [3] generates FOA audio but assumes static, context-free
sound sources, while Sonic4D [4] estimates 3D positions but out-
puts only stereo audio, reducing immersion. These approaches typ-
ically neglect how environmental factors, such as moving objects,
walls, or furniture, affect propagation through occlusion, reflec-
tions, and reverberation. This omission results in sound fields that
lack physical grounding and fail to adapt convincingly to user ori-
entation.
To address these limitations, we present DynFOA, a geometry-
and material-aware FOA generation framework for 360-degree
videos. Our method explicitly reconstructs the scene’s 3D geom-
etry from monocular 360-degree video using a pipeline consisting
of sound source detection, dense depth estimation, semantic seg-
mentation, and 3D Gaussian Splatting (3DGS) reconstruction. The
resulting geometry is augmented with per-surface material proper-
ties to derive acoustic descriptors, including occlusion masks, re-
flection paths, and frequency-dependent reverberation times. These
descriptors serve as conditioning signals for a diffusion-based FOA
generator, enabling audio rendering that responds dynamically to
both the spatial structure of the scene and the listener’s head orien-
tation. By conditioning the denoising process on real scene geome-
try, DynFOA produces physically consistent spatial audio that pre-
serves directionality, distance cues, and environmental characters.
Our evaluations demonstrate that DynFOA not only achieves su-
perior spatial accuracy, acoustic fidelity, and distribution matching
compared to existing methods, but also provides a more engaging
user experience.
Our key contributions are summarized as follows:
• Visual-Acoustic Scene Reconstruction: We introduce a vi-
sual processing pipeline that reconstructs scene geometry
and materials from 360-degree video, enabling physically
grounded modeling of occlusion, reflections, and reverbera-
tion effects in spatial audio rendering.
• Conditional Diffusion for FOA Generation: We design a
conditional diffusion model operating in the latent FOA do-
main, where denoising steps are guided by geometry- and
material-aware acoustic descriptors, allowing spatial audio to
adapt naturally to listener orientation and viewpoint changes.
• Special Datasets: We create a comprehensive dataset Dyn360
for complex acoustic scenes with occlusion, reflections, and
reverberation, which contains 600 video scenes and three sub-
datasets of their content in Sec. 4.1.1.
• Comprehensive Evaluation: We establish an extensive eval-
uation framework specifically targeting Dyn360 as a compre-
hensive benchmark, and experiments demonstrated that our
DynFOA achieved state-of-the-art performance in both quan-
titative and qualitative analysis.
By uniting physically grounded scene reconstruction with
diffusion-based generative modeling, DynFOA advances spatial au-
dio rendering beyond purely perceptual alignment toward true au-
diovisual physical coherence, offering a new pathway for immer-
sive media production in VR and cinematic 360-degree experi-
ences.
2
RELATED WORK
2.1
Spatial Audio Representations and Rendering
Spatial audio reproduction in VR has long relied on Ambisonics,
which encodes sound fields in the spherical harmonic domain [5].
First-Order Ambisonics (FOA) strikes a balance between spatial
resolution and computational efficiency, while higher-order Am-
bisonics improves angular precision at a higher computational cost.
The mathematical foundation of spherical harmonic decomposition
allows for 3D sound field capture and reconstruction at varying
levels of accuracy, making Ambisonics particularly well-suited for
VR, where perceptual fidelity must be balanced with real-time pro-
cessing constraints.
For spatial audio rendering, Head-Related Transfer Functions
(HRTFs) are essential, providing the necessary filtering to simu-
late spatial localization. Standardized resources, such as KEMAR
[6] and CIPIC [7], along with exchange formats like SOFA [8],
form the foundation for HRTF-based rendering. However, indi-
vidualization remains a challenge due to anthropometric variabil-
ity, which affects localization and externalization of sounds. Re-
cent works have addressed this issue through both direct measure-
ments and computational models. Platforms for acoustic simula-
tion, such as SoundSpaces [9, 10], offer 3D scanned environments
with physically-grounded propagation models, enabling systematic
evaluation and cross-modal learning. These platforms emphasize
the importance of jointly modeling visual and acoustic cues for bet-
ter audio-visual scene understanding. In the context of 360-degree
video, early work [11] generated FOA from mono audio, using
panoramic cues. OmniAudio [3] formalized the 360V2SA task, but
with the assumption of static sources and negligible occlusion or
reverberation. Sonic4D [4] reconstructed dynamic 3D trajectories
for viewpoint-adaptive binaural rendering, though it did not pro-
duce FOA. More recent geometry-aware methods [12, 13] leverage
depth cues, but they typically focus only on source distance.
However, these existing works have not sufficiently considered
the influence of scene geometry and materials, which affect occlu-
sion, reflections, and reverberation, in the generation of FOA for
complex scenes involving dynamic sound sources.
2.2
Cross-Modal Learning for Sound Localization and
Separation
Cross-modal learning leverages the synergy between different
modalities: visual signals help constrain sound localization and dis-
ambiguate overlapping events, while audio assists in grounding vi-
sual understanding. This synergy is particularly crucial for immer-
sive spatial audio, where multiple sound sources and complex envi-
ronments necessitate robust sound separation. A significant body of
work has explored cross-modal learning for sound localization and
separation using visual cues. Weakly supervised methods align ob-
jects with sounds [14, 15], while co-separation frameworks learn to
disentangle multiple overlapping sound sources. Other approaches
focus specifically on speech, such as speaker-independent audio-
visual separation [16] and lip-synchronized speech extraction [17].
Large-scale datasets like AudioSet [18], VGGSound [19], and MU-
SIC [20] have significantly accelerated these advances by providing
diverse audio-visual training resources.
Our framework builds on these principles but extends them
to FOA generation. By integrating monocular 360-degree vision
with geometry- and material-aware acoustic descriptors, we enable
sound localization and separation in acoustically complex environ-
ments, where traditional cross-modal methods often fall short.
2.3
Visually Guided Spatialization for 360-Degree Video
Recent advancements have focused on vision-guided audio spatial-
ization. Early approaches addressed mono-to-binaural conversion
using neural networks [21], while later works shifted toward stereo
or pseudo-binaural generation [22]. Methods like Points2Sound
2

<!-- page 3 -->
incorporated geometry and motion cues to improve spatialization
[23], and newer pipelines introduced explicit depth estimation or
scene reconstruction for greater accuracy. Despite these develop-
ments, many approaches still oversimplify acoustic propagation,
often assuming free-field conditions or modeling only distance at-
tenuation. To address this, OmniAudio [3] and Sonic4D [4] made
significant progress by targeting panoramic or dynamic sources.
However, they did not account for FOA outputs or more complex
propagation effects, such as reverberation. Head-tracked playback
pipelines further demonstrated the necessity of synchronizing au-
dio rotation with the listener’s orientation, but they remain limited
in handling occlusion and material diversity.
To overcome these limitations, we leverage 360-degree scene
reconstruction with per-surface material estimation. By modeling
occlusion, reflections, and reverberation, we condition FOA gen-
eration on physically meaningful descriptors, enabling both head-
tracked adaptability as well as geometry- and material-aware spatial
realism.
2.4
Diffusion Models for High-Fidelity Audio Generation
Generative models have recently revolutionized audio synthesis.
Diffusion models, such as WaveGrad [24], DiffWave [25], and
AudioGen [26], have set new benchmarks in waveform genera-
tion. Spatial audio frameworks based on diffusion, such as Diff-
SAGe [27] and ImmerseDiffusion [28], demonstrate strong poten-
tial, though they focus primarily on perceptual fidelity rather than
ensuring physical scene consistency. Meanwhile, autoregressive
models like SoundStorm [29] and controllable music generators
[30] offer sequence-level control but struggle with long-term coher-
ence. Video-to-audio pipelines, such as ViSAGe [31] and MMAu-
dio [32], integrate motion and semantic cues but rarely address FOA
generation or physically grounded propagation.
DynFOA addresses these gaps by embedding geometry- and
material-aware descriptors into a conditional diffusion pipeline for
FOA generation. This approach enables us to combine the high-
fidelity synthesis capabilities of diffusion models with explicit mod-
eling of occlusion, reflections, and reverberation, advancing gener-
ative spatial audio from perceptual plausibility to physical consis-
tency.
3
METHODOLOGY
3.1
Problem Definition
The objective of this work is to enable physically grounded
and perceptually coherent 4D immersive experiences from 360-
degree videos by learning to generate scene-aware spatial audio.
Prior approaches to spatial audio rendering often rely on simpli-
fied acoustic assumptions, neglecting critical aspects such as dy-
namic sound sources, concurrent source interactions, and propa-
gation effects including occlusion, reflections, and reverberation.
Our framework, DynFOA, directly addresses these challenges by
learning from multimodal cues—visual appearance, 3D geome-
try, and material properties—to synthesize first-order ambisonics
(FOA) that faithfully reflect the physical structure and acoustic con-
ditions of the scene [33, 26].
Formally, we cast this task as learning a mapping:
fθ : (V,G,M,R,o) 7→S4D,
(1)
where V is the 360-degree video stream, G denotes the recon-
structed 3D scene geometry, M represents per-surface material
properties, R encodes reverberation and reflection parameters, and o
specifies the listener’s head orientation. The learnable function fθ,
parameterized by θ, integrates these modalities to generate a multi-
modal 4D representation S4D, in which spatially aligned audio and
visual cues jointly define the immersive experience.
Solving this problem requires addressing a sequence of coupled
sub-tasks across both the visual and audio domains. On the vi-
sual side, the model must (i) detect and localize sound-emitting
and non-emitting objects, (ii) estimate depth, (iii) perform semantic
segmentation, and (iv) reconstruct a geometry- and material-aware
3D representation using techniques such as 3D Gaussian Splatting
(3DGS) [34]. On the audio side, the model must (i) extract di-
rectional cues from FOA channels, (ii) encode them into a latent
representation, and (iii) model complex propagation phenomena in-
cluding occlusion, reflections, and reverberation [33].
The key challenge lies not only in localizing sound sources
but also in handling multiple, dynamic sources within acoustically
complex environments [35].
By grounding audio generation in
geometry- and material-aware acoustic descriptors, our model cap-
tures both static and dynamic elements of the scene. This enables
real-time adaptation to source motion, ensuring accurate localiza-
tion, separation, and a physically consistent audio field that reflects
the spatial relationships inherent in 360-degree visual scenes [36].
3.2
Overview
Figure 2 illustrates the proposed DynFOA framework, which gen-
erates dynamic and physically consistent first-order ambisonics
(FOA) from 360-degree video. The model consists of three mod-
ules: a Video Encoder, an Audio Encoder, and a Conditional
Diffusion Generator. The Video Encoder (Sec. 3.3) reconstructs
3D scene geometry and material properties from 360-degree video.
It detects and tracks dynamic sound sources, estimates depth, and
applies semantic segmentation, producing geometry- and material-
aware acoustic descriptors such as occlusion, reflections, and re-
verberation. The Audio Encoder (Sec. 3.4) processes FOA sig-
nals into geometry-aware embeddings. Through spectral decom-
position, spherical harmonic transformation, and spatial mapping,
it captures directional cues, attenuation, and material absorption,
while integrating saliency and reverberation features for consis-
tency with the visual scene. At the core, the Conditional Diffu-
sion Generator (Sec. 3.5) fuses video- and audio-derived features
via a multi-condition encoder. Geometry, material, and propaga-
tion cues—along with descriptors of dynamic and multiple sound
sources—guide a U-Net denoiser to synthesize FOA that are both
physically grounded and perceptually realistic. During inference,
the generated FOA is rotated according to the listener’s head ori-
entation and rendered binaurally with HRTFs, enabling real-time,
head-tracked playback with immersive spatial audio.
3.3
Video Encoder
The Video Encoder extracts spatial and semantic cues from 360-
degree video to support realistic sound propagation modeling and
synchronized spatial audio rendering [37].
It operates in three
stages: (1) sound source localization and depth estimation, (2) se-
mantic segmentation and scene reconstruction, and (3) feature ex-
traction and multi-modal fusion.
3.3.1
Sound Source Localization and Depth Estimation
The encoder first detects and localizes sound-emitting objects in
the scene [38]. Each source i is assigned a bounding box ˆbi and an
activity score ˆyi, optimized by:
Lob j = ∑
i

∥bi −ˆbi∥2 +(yi −ˆyi)2
,
(2)
where bi and ˆbi denote the ground truth and predicted spatial pa-
rameters, while yi ∈{0,1} and ˆyi ∈[0,1] represent the true and pre-
dicted activity status. This ensures accurate detection and temporal
tracking of dynamic sound sources.
Depth estimation then back-projects pixel-level depth into 3D
points:
3

<!-- page 4 -->
Figure 2: Overview of Our DynFOA. The model features a three-core architecture: audio encoder, video encoder, and conditional diffusion
generator. The audio encoder enhances FOA audio robustness against occlusion, reflections, and reverberation through dynamic sound source
processing. The video encoder reconstructs video information to support immersive FOA experiences. The conditional diffusion integrates
scene context for deep FOA encoding and multimodal fusion, enabling real-time, high-fidelity immersive FOA rendering in scenes with
occlusion, reflections, and reverberation.
p(u,v) = D(u,v)[cos(θ)cos(φ), sin(θ), cos(θ)sin(φ)]T ,
(3)
where (u,v) are image coordinates, D(u,v) is depth, and θ,φ are the
corresponding elevation and azimuth angles. The resulting point
cloud serves as the geometric basis for acoustic modeling. To re-
construct the full scene, a hybrid approach combines Truncated
Signed Distance Functions (TSDF) for large-scale structures with
3D Gaussian Splatting (3DGS) for fine details [39, 40], enabling
accurate modeling of occlusion, reflections, and reverberation.
3.3.2
Semantic Segmentation and Scene Reconstruction
The encoder further applies semantic segmentation to classify scene
elements (e.g., walls, floors, furniture). Each class is mapped to
frequency-dependent acoustic material properties [41, 42], enrich-
ing the reconstructed geometry with absorption and reflection pa-
rameters. These semantic and geometric cues are integrated into a
3D scene model [43], enabling simulation of occlusion, reflections,
and reverberation based on both structure and material characteris-
tics. This ensures that environmental effects such as sound block-
ing, scattering, and decay are faithfully captured.
3.3.3
Feature Extraction and Fusion
Finally, spatial and temporal features are extracted using Convo-
lutional Neural Networks (CNNs) and Recurrent Neural Networks
(RNNs) [44]. These video-derived features are fused with audio
representations to jointly model scene dynamics and acoustic con-
ditions. The resulting multimodal features support real-time track-
ing of dynamic sound sources while incorporating propagation ef-
fects (occlusion, reflections, and reverberation), thereby enabling
perceptually coherent and immersive spatial audio rendering.
3.4
Audio Encoder
The Audio Encoder processes FOA signals to extract directional,
spectral, and temporal cues that align with the geometry- and
material-aware representations produced by the Video Encoder. By
coupling FOA features with reconstructed scene properties, it en-
ables context-aware spatialization consistent with the physical lay-
out and acoustic characteristics of the environment [45, 46].
3.4.1
FOA Extraction and Normalization
We begin by extracting the four FOA channels, W, X, Y, and Z,
which jointly represent omnidirectional and directional components
of the sound field. To stabilize training and ensure consistent scal-
ing across channels, z-score normalization is applied by subtracting
the mean and dividing by the standard deviation of each channel.
This reduces magnitude imbalance and provides a robust founda-
tion for downstream feature learning.
From the normalized channels, a CNN extracts compact repre-
sentations of spectral and directional patterns. Convolutions over
the time–frequency domain capture harmonic content and inter-
channel correlations, while stacked layers aggregate these into
higher-level spatial features [44]. The resulting FOA embeddings
form the basis for geometry-aware mapping and materially consis-
tent spatial audio generation.
3.4.2
Spatial and Directional Mapping
To incorporate structural priors, we modulate FOA features with
geometric distance and material-dependent absorption [45]. This
accounts for sound attenuation and redirection during propagation:
Apath = ∏
j
(1−αm j)·e−γd,
(4)
where αm j is the absorption coefficient of the j-th material along
the path, d is the propagation distance, and γ is the air attenuation
4

<!-- page 5 -->
factor. This formulation supports the modeling of occlusion, early
reflections, and late reverberation.
Directional information is further captured by projecting FOA
features onto a spherical harmonic basis, yielding a compact repre-
sentation of spatial energy distributions [47]. This transformation
reinforces alignment between audio embeddings and reconstructed
scene geometry, enabling accurate reasoning about sound propaga-
tion across directions.
3.4.3
Saliency, Reverberation, and Output
To highlight perceptually relevant cues, an attention mechanism
modulates FOA features with visual saliency and acoustic context
[46]:
at = σ(Watt[Fenc;Mvis]+batt),
(5)
where [Fenc;Mvis] concatenates encoded FOA features and visual
saliency maps, Watt and batt are learnable parameters, and σ is the
sigmoid function. This selective amplification refines geometry-
and material-aware features, enhancing the simulation of occlusion,
reflections, and reverberation.
Spatial realism is further enriched by augmenting FOA features
with late reverberation profiles that capture long-range energy de-
cay and diffusion [42].
Estimated from reconstructed geometry
and material properties, these profiles complement direct sound and
early reflections, yielding acoustically consistent reverberation pat-
terns.
The Audio Encoder ultimately produces a geometry- and
material-aware FOA embedding that is decoded into the four out-
put channels. This representation preserves directional, spectral,
and temporal consistency while remaining aligned with scene ge-
ometry, ensuring that the rendered spatial audio is both physically
grounded and perceptually coherent.
3.5
Conditional Diffusion Generator
Inspired from recent spatial audio diffusion frameworks such as
Diff-SAGe [27] and ImmerseDiffusion [28], we employ a condi-
tional diffusion model to synthesize FOA that remain consistent
with reconstructed scene geometry and material properties. Op-
erating in the latent FOA domain, the model integrates structural
and acoustic cues—including occlusion, reflections, and reverber-
ation—while accounting for dynamic and multiple sound sources.
Starting from a noisy latent representation, a U-Net denoiser pro-
gressively reconstructs clean FOA signals over T timesteps. Train-
ing follows the denoising diffusion probabilistic model (DDPM)
objective:
Ldiff = Ex0,ε∼N (0,1),t
h
∥ε −εθ (xt,t,c)∥2
2
i
,
(6)
where xt is the noisy FOA latent at timestep t, ε denotes Gaussian
noise, εθ is the U-Net denoiser, and c is a conditioning vector that
aggregates scene and propagation features.
3.5.1
Conditioning on Geometry and Material Properties
Reconstructed geometry and material attributes provide the founda-
tion for physically grounded synthesis. The 3D mesh encodes struc-
tural layout and surface orientation, while material properties spec-
ify frequency-dependent absorption coefficients [42]. Embedding
these features allows the model to account for attenuation, diffrac-
tion, and spatial filtering effects, thereby ensuring that the generated
FOA is consistent with the reconstructed scene. Such conditioning
is particularly important in scenarios involving dynamic or multiple
sound sources, where accurate modeling of energy interaction with
the environment is essential.
3.5.2
Conditioning for Occlusion, Reflections, and Rever-
beration
To capture realistic propagation effects, the model is further con-
ditioned on occlusion, reflections, and reverberation.
Occlusion
features are derived from visibility analysis between listener and
sources, modulated by material absorption. Early reflections are
estimated by tracing geometric paths, providing echo-like cues
that enhance spatial depth.
Reverberation is represented using
frequency-dependent T60(f) curves, which describe late decay
characteristics [28]. Together, these cues enrich the conditioning
stream, enabling FOA synthesis that incorporates both direct sound
and its environmental response.
3.5.3
Multi-Condition Encoder and Cross-Modal Fusion
All conditional features are projected into a shared latent space be-
fore being injected into the diffusion U-Net. Modulation layers and
cross-attention mechanisms fuse geometry, material, and propaga-
tion cues with descriptors of dynamic and multiple sound sources
[48]. This cross-modal integration guides the denoising trajectory,
ensuring that the generated FOA respects physical propagation con-
straints while maintaining perceptual consistency across time and
sources.
3.5.4
Runtime Rendering and Head-Tracking
At inference, the diffusion model generates FOA conditioned on the
reconstructed scene and dynamic context. The synthesized FOA
signals are rotated according to the listener’s head orientation to
maintain spatial alignment under head tracking [46, 45]. Finally,
FOA are rendered to binaural signals using head-related transfer
functions (HRTFs). This runtime process produces immersive spa-
tial audio that adapts seamlessly to listener movement and complex
multi-source environments.
4
EXPERIMENT
4.1
Experiment Setup
4.1.1
Dataset Usage and Construction
Our experiment is conducted based on the YT360 and Sphere360
datasets [3], and uses a bilingual Chinese and English keyword se-
mantic filtering algorithm combined with a manual review mech-
anism to construct our Dyn360 dataset. This dataset contains 600
strictly screened high-quality video clips that have been optimized
to accurately reconstruct spatial audio in complex acoustic environ-
ments. All samples are normalized to 10 seconds in length, encoded
in H.264 format, with a resolution of at least 720p and a stable
frame rate of 30fps. The audio data is stored at a 16kHz sampling
rate, 16-bit depth, and 4-channel format to generate high-quality
FOA. All of the 600 clips are used for our quantitative evaluation
reported later in Sec. 4.3.
To conduct more targeted research, we employed a combination
of keyword identification and content-based filtering to categorize
the source material into three distinct sub-datasets. First, as differ-
ent geometric shapes exert distinct influences on the quality of FOA
generation, we constructed a geometry sub-dataset (Geometry) by
extracting video segments containing geometric elements, includ-
ing basketballs and boxes in the same scene (corresponding to cir-
cles and squares with their respective different materials). This sub-
dataset is designed to investigate how different geometries affect
spatial audio rendering, and it contains 365 clips encoded in H.264
format at 640×360 resolution and 18.70 fps. Second, we established
the move sound source sub-dataset (MoveSource) through manual
curation of dynamic scenes featuring moving objects such as ve-
hicles and pedestrians, This sub-dataset provides controlled condi-
tions for analyzing occlusion and material-dependent propagation
effects in dynamic environments, comprising 128 clips in H.264
format at 640×360 resolution and 29.01 fps. Third, we developed
5

<!-- page 6 -->
the multi-sound source sub-dataset (MultiSource) by aggregating
complex acoustic scenes containing two or more distinct sound
sources. This sub-dataset emphasizes challenges such as reflec-
tions and reverberation in multi-source scenarios, yielding 107 clips
in H.264 format at 640×360 resolution and 12.45 fps. All source
videos underwent standardized preprocessing to comply with our
Dyn360 specifications, including upsampling to a minimum resolu-
tion of 720p, frame rate normalization to 30fps, and segmentation
into 10-second clips, thereby enabling comprehensive evaluation of
our model’s adaptability across diverse scene complexities includ-
ing occlusion, reflections, and reverberation.
4.1.2
Evaluation Metrics
We evaluated the FOA generation quality from our DynFOA and
baselines along four different dimensions: spatial accuracy, acous-
tic fidelity, audio distribution matching, and user study protocol.
Our evaluation metrics is specifically designed to validate the ef-
fectiveness of our DynFOA in handling complex acoustic scenarios,
including occlusion, reflections, and reverberation.
Spatial accuracy evaluates the directional accuracy of the gener-
ated FOA using Direction of Arrival (DOA) estimation [49], which
measures the angular accuracy of the predicted sound source posi-
tions relative to the true source positions in 3D space.
Acoustic fidelity assesses different model’s ability to cap-
ture complex acoustic environments, using Signal-to-Noise Ratio
(SNR) [50] to measure audio clarity in the presence of background
noise, and Early Decay Time (EDT) [51] to assess reverberation
characteristics in various acoustic spaces, following established
practices in spatial audio evaluation.
Audio distribution matching evaluates the similarity of feature
distributions between real and generated FOA under our DynFOA
and baselines. According to existing experimental progress, we
compute the Fr´echet Distance (FD) [32, 3] using a specialized spa-
tial audio feature extractor. Furthermore, we introduced Short-Time
Fourier Transform Error (STFT) [52] to measure spectral recon-
struction accuracy, Scale-Invariant Signal-to-Distortion Ratio (SI-
SDR) [53] to assess signal separation quality, and Kullback-Leibler
(KL) [3] divergence to evaluate the statistical similarity between
generated and reference audio distributions.
User study protocol was designed to conduct subjective eval-
uation using Mean Opinion Score (MOS-SQ) and Audio-Visual
Alignment Fidelity (MOS-AF) [3] to assess the subjective user ex-
perience. Human evaluators rated the realism and synchronization
of the generated FOA within the complex scenes including occlu-
sion, reflections and reverberation.
4.1.3
Baselines
Since our work focuses on extracting scene information through 3D
Gaussian reconstruction techniques and diffusion models to gener-
ate high-quality FOA from 360-degree videos, we construct the fol-
lowing benchmarks for comprehensive comparison: (1) Diff-SAGe
[27], a diffusion-based spatial audio generation model that uses a
diffusion process for FOA audio synthesis. (2) MMAudio [32] with
spatialization adaptation, this combined method integrates the state-
of-the-art multimodal audio generation model MMAudio [32] and
an audio spatialization component. The spatialization component
uses spatial angle estimation to convert the generated mono/stereo
audio into a first-order surround sound FOA format. (3) Omni-
Audio [3], the original OmniAudio framework for converting 360-
degree videos into spatial audio, which is our main benchmark for
spatial audio synthesis (without using 3D Gaussian scene recon-
struction). (4) ViSAGe [31] , a traditional model designed to gen-
erate spatial audio from video input, focusing on the spatial au-
dio generation task and performing well on directional audio syn-
thesis. To ensure fair and consistent comparisons, we train and
test all the models using proportional splits based on the Dyn360
dataset. Each baseline model takes a panoramic 360-degree video
as its primary input, while the combined model accepts optional
text prompts when available.
4.2
Quantitative Results
We conducted a comprehensive evaluation of our DynFOA against
baselines, as described in Sec. 4.1.3, which spans three specialized
acoustic scenarios about geometry, move sound source, and multi-
sound source within the Dyn360 dataset. The evaluation employs
both objective metrics measuring spatial accuracy, acoustic fidelity,
and audio distribution matching, alongside user study protocol to
validate the effectiveness of our DynFOA. All quantitative Results
are presented in Table 2.
Geometry-Material Localization Accuracy. DynFOA demon-
strates superior spatial accuracy in geometry-aware and material-
sensitive localization using DOA estimation across all 600 clips.
In the Geometry sub-dataset focusing on geometric acoustic envi-
ronments with various material properties, our DynFOA achieves
exceptional spatial precision with a DOA error of only 0.12 com-
pared to other baselines. The MoveSource sub-dataset results reveal
even stronger performance in move sound source scenarios (DOA:
0.08), while maintaining robust accuracy in the MultiSource sub-
dataset (DOA: 0.12). This consistency contrasts sharply with base-
line methods, where ViSAGe shows significant degradation across
scenarios (DOA ranging from 0.45 to 0.54). The consistently low
DOA errors demonstrate the effectiveness of our 3D Gaussian scene
reconstruction in capturing precise spatial relationships between ge-
ometric structures and material-dependent acoustic properties, with
DynFOA maintaining superior localization accuracy even in com-
plex scenarios where competing methods like Diff-SAGe struggle
with errors exceeding 0.36.
Reflections and Reverberation Modeling. The SNR and EDT
measurements reveal DynFOA’s exceptional acoustic fidelity in
modeling reflections and reverberation within complex acoustic en-
vironments. Specifically, DynFOA consistently achieves the high-
est SNR values across all scenarios (18.37-19.92 dB) while ef-
fectively handling acoustic reflections from various surfaces, and
maintains optimal reverberation characteristics with the lowest
EDT scores (0.03-0.04) to accurately assess reverberation decay
in different acoustic spaces. This dual excellence indicates effec-
tive capture of both direct sound clarity and complex environmental
acoustic phenomena including surface reflections and room rever-
beration, significantly outperforming methods like ViSAGe which
show substantially lower SNR performance (9.74-12.24 dB) and
struggle with reflection modeling.
Occlusion-Enhanced Distribution Matching. DynFOA excels
in occlusion-aware distribution matching by preserving the statis-
tical and spectral characteristics between real and generated FOA
under various occlusion conditions. Our method achieves superior
FD scores across all scenarios (0.06-0.09) using specialized spatial
audio feature extraction that accounts for sound occlusion effects,
indicating excellent distribution matching between generated and
reference audio even when sound sources are partially or fully oc-
cluded. The consistently low STFT errors (0.11-0.15) demonstrate
precise spectral reconstruction accuracy under occlusion scenarios,
while high SI-SDR values (14.47-15.58) assess superior signal sep-
aration quality in the presence of occluding objects, and optimal
KL divergence confirms statistical similarity between generated and
reference audio distributions across different occlusion patterns.
User Study Results. Our subjective experiment consists of 50
people with an average age of about 24 and an age standard devia-
tion of about 1.3. Of the respondents, 73% are male and 27% are
female. Human evaluation confirms DynFOA’s perceptual advan-
tages through consistently high MOS scores. Our method achieves
MOS-SQ ratings approaching ground truth levels (4.34-4.38 vs
GT: 4.58-4.67) across all scenarios, with particularly strong audio-
6

<!-- page 7 -->
Table 1: DynFOA’s performance is compared against baselines on three specific sub-datasets of the Dyn360. Results are compared for
Geometry, MoveSource, and MultiSoucre. The evaluation covers technical accuracy metrics: DOA, SNR, EDT, FD, STFT, SI-SDR, and
KL, as well as user study using MOS-SQ and MOS-AF. Ground truth values provide a reference standard for subjective evaluation. Arrows
indicate performance direction: ↓indicates better performance for lower metrics, ↑indicates better performance for higher metrics.
Model
DOA↓
SNR↑
EDT↓
FD↓
STFT↓
SI-SDR↑
KL↓
MOS-SQ↑
MOS-AF↑
Geometry
GT
-
-
-
-
-
-
-
4.64
4.50
DynFOA (ours)
0.12
18.37
0.03
0.09
0.15
15.02
0.27
4.36
4.00
Diff-SAGe
0.33
12.12
0.12
0.24
0.49
9.79
0.59
3.12
2.72
MMAudio+spatialization
0.24
13.20
0.09
0.18
0.35
10.96
0.47
3.42
3.11
OmniAudio
0.18
16.41
0.05
0.12
0.20
12.34
0.34
3.61
3.76
ViSAGe
0.45
9.74
0.17
0.35
0.60
7.56
0.78
2.56
2.32
MoveSource
GT
-
-
-
-
-
-
-
4.67
4.44
DynFOA (ours)
0.08
19.92
0.03
0.06
0.11
15.58
0.17
4.38
4.17
Diff-SAGe
0.36
12.93
0.12
0.25
0.40
10.65
0.56
3.11
3.16
MMAudio+spatialization
0.23
15.69
0.07
0.16
0.29
12.94
0.42
3.68
3.26
OmniAudio
0.15
18.13
0.04
0.09
0.18
13.75
0.28
3.92
3.84
ViSAGe
0.51
12.24
0.18
0.38
0.60
8.92
0.72
2.67
2.51
MultiSource
GT
-
-
-
-
-
-
-
4.58
4.49
DynFOA (ours)
0.12
18.90
0.04
0.08
0.12
14.47
0.19
4.34
4.19
Diff-SAGe
0.39
12.16
0.10
0.28
0.41
9.86
0.50
3.11
3.18
MMAudio+spatialization
0.26
14.74
0.06
0.18
0.28
11.96
0.38
3.69
3.32
OmniAudio
0.18
16.99
0.05
0.12
0.19
12.87
0.26
4.01
3.89
ViSAGe
0.54
11.14
0.14
0.42
0.59
7.95
0.66
2.64
2.55
visual alignment scores (MOS-AF: 4.00-4.19). These results val-
idate that our technical improvements translate effectively to en-
hanced user experience, maintaining critical synchronization even
under varying acoustic complexities.
4.3
Qualitative Evaluation
For a targeted comparison, we have selected two representative
scenes, namely a piano scene with complex sound source charac-
teristics and a ballroom scene with a variety of different timbres
and tones, which can just prove the powerfulness of our DynFOA.
In this qualitative evaluation, we used spectral analysis to compare
the generated FOA quality of our DynFOA with that of existing
baseline methods as outlined in Sec. 4.1.3. The results are shown
in Figure 3 and Figure 4. Compared with baselines, which ex-
hibit mid–high frequency attenuation, loss of low-frequency spa-
tial correlation, and incomplete harmonic reconstruction, DynFOA
achieves substantially higher fidelity in acoustic scene modeling by
integrating panoramic visual cues, thereby producing spectrograms
that more closely align with the ground truth distribution both in
frequency coverage and statistical similarity. In FOA generation,
DynFOA further demonstrates superior multi-channel spatial cod-
ing capacity: comparative analysis across the W, X, Y, and Z chan-
nels reveals stronger preservation of directional cues, more accu-
rate sound-field reconstruction, and greater robustness to complex
acoustic phenomena. Even under challenging environmental con-
ditions such as significant source occlusion, room reflections, and
high reverberation levels, our DynFOA maintains stable correla-
tions between the four FOA channels and precise audio spatial lo-
calization, demonstrating its adaptability and capabilities in diverse
and complex acoustic environments. Furthermore, by incorporat-
ing a conditional diffusion model into the generative framework,
DynFOA more effectively captures the inherent randomness and
fine-grained texture of real-world audio signals than determinis-
tic methods. This improvement is intuitively reflected in the shifts
between different frequency spectra in the spectrogram, which ex-
hibits richer harmonic detail, smoother frequency transitions, and
a noise structure that better aligns with the statistical properties of
natural sound and human perception, ultimately enhancing the au-
dio’s realism, immersion, and three-dimensionality.
4.4
Ablation Studies
Impact of Scene Information. In the first ablation study, we com-
pared the audio-only variant with the gradual addition of different
visual priors, as shown in Table 2. From audio-only to audio +
visual detection, DOA and EDT both decrease notably, indicating
that visual detection helps reduce localization error and improve
temporal stability. Further incorporating the depth prior contin-
ues to lower DOA and EDT while slightly reducing frequency-
domain distortion, showing that depth information effectively en-
hances spatial consistency without harming SNR. Finally, introduc-
ing the geometric scene achieves the lowest DOA and EDT values
and the smallest FD, representing the most stable spatial pointing
and the least frequency-domain artifacts. These results confirm that
gradually adding scene priors improves spatial accuracy, temporal
coherence, and frequency-domain fidelity, bringing the generated
audio closer to the reference.
Impact of Diffusion Model. The second ablation study exam-
ined the effects of diffusion modeling and directional condition-
ing. As presented in Table 3, non-diffusion regression produces
relatively high SNR but suffers from large DOA and EDT errors
7

<!-- page 8 -->
Figure 3: The piano performance performed by a singer and pianist in an indoor setting. Due to the occlusion and distribution of objects,
the environment exhibits significant occlusion, reflections, and reverberation characteristics. Existing methods often reduce the clarity and
immersiveness of FOA. However, our DynFOA effectively accounts for factors affecting audio quality in this complex scene, accurately
capturing spatial cues while preserving the natural timbre of instruments and vocals. Compared to existing baseline methods, our DynFOA
achieves more realistic sound reproduction, delivering a more immersive experience.
Table 2: Gradually introducing scene information reduces DOA and
EDT errors while improving frequency-domain stability.
Model
DOA↓
SNR↑
EDT↓
FD↓
Audio-Only
0.320
-7.08
0.048
1.373
Audio+Visual
0.280
-7.08
0.045
1.370
Audio+Visual+Depth
0.260
-7.08
0.044
1.365
Audio+Visual+Depth+Geo
0.240
-6.96
0.042
1.360
as well as higher FD, highlighting the limitations of determinis-
tic regression in maintaining spatial stability. Introducing uncondi-
tional diffusion changes this behavior: although the SNR decreases
due to stochastic sampling, temporal noise modeling reduces DOA
and EDT errors, leading to more coherent cross-frame trajectories.
Adding conditional diffusion with video-derived cues further im-
proves DOA and FD, demonstrating that conditioning primarily
enhances spatial coherence and frequency-domain robustness. In-
creasing the sampling steps achieves the lowest DOA and EDT er-
rors and further reduces FD, indicating that larger step sizes mainly
refine stability and fidelity rather than altering the underlying mech-
anism.
Table 3: Ablation on regression VS diffusion modeling, condition-
ing, and sampling steps.
Model
DOA↓
SNR↑
EDT↓
FD↓
Non-Diffusion
0.320
-0.7082
0.048
1.420
Unconditional Diffusion
0.280
-0.7082
0.046
1.380
Conditional Diffusion
0.260
-6.960
0.044
1.365
Conditional Diffusion+Steps
0.240
0.520
0.042
1.350
Impact of Model Scale. We next analyze the effect of model
scale on FOA quality, as presented in Table 4. The minimal model
offers the fastest inference but exhibits larger DOA errors, weaker
SNR, and noticeable frequency distortion. Scaling up to the small
and medium models steadily improves directional stability, miti-
gates frequency distortion, and enhances spectral fidelity, indicat-
ing that larger models better capture spatial and temporal cues.
The large model further reduces DOA drift and produces smoother,
more consistent audio. Finally, the maximal model achieves the
lowest DOA error and distortion, delivering the most stable and
high-quality FOA, albeit at the cost of computational efficiency.
Overall, these results demonstrate a clear positive correlation be-
tween model scale and FOA quality. Larger models not only cap-
ture finer-grained spatial geometry but also more faithfully recon-
struct material-dependent acoustic behaviors such as occlusion, re-
flections, and reverberation, leading to superior distribution fidelity
and perceptual scene realism.
Table 4: Effect of model scale on FOA generation quality.
Model
DOA↓
SNR↑
EDT↓
FD↓
Minimal
0.310
-7.10
0.044
1.42
Small
0.290
-6.80
0.043
1.35
Medium
0.270
-6.20
0.041
1.25
Large
0.255
-5.80
0.040
1.15
Maximal
0.240
-5.40
0.039
1.05
4.5
Limitations and Future Work
While our DynFOA has made significant progress in spatial au-
dio generation, its performance is still limited by certain acoustic
environments. Beyond common factors such as occlusion, reflec-
tions, and reverberation, more explicit modeling of scene-specific
8

<!-- page 9 -->
Figure 4: The dance hall events involve multiple simultaneous sound sources and complex spatial dynamics. This environment presents
numerous challenges, such as overlapping vocalizations, dynamic motion of sound sources, and highly variable reverberation patterns. Our
DynFOA effectively separates these concurrent audio elements, reconstructing the scene’s spatial distribution with superior accuracy while
maintaining spectral consistency. This demonstrates our DynFOA exceptional robustness in handling highly complex real-world acoustic
conditions, outperforming baseline models.
sound source conditions is needed to further improve the fidelity
and robustness of the generated audio. Furthermore, differences
in material properties and sound propagation media pose consider-
able challenges. For example, underwater and terrestrial environ-
ments exhibit fundamentally different acoustic propagation mech-
anisms, which were not fully explored in this study. Current esti-
mates of material properties based on semantic segmentation only
provide approximate representations of acoustic properties and fail
to capture complex, frequency-dependent surface effects that sig-
nificantly influence sound behavior. Furthermore, the experiments
and evaluations in this study were primarily conducted in controlled
indoor environments with limited environmental variation. Because
model training and inference are highly sensitive to environmental
variations, our evaluations still face challenges in generalizing to
outdoor or uncontrolled acoustic environments.
Future work will address the current research’s inadequate mod-
eling of material properties, propagation media, and adaptability
to complex environments. We will further explicitly incorporate
acoustic factors such as occlusion, reflections, and reverberation,
and expand our approach to outdoor and cross-media scenarios. By
optimizing our diverse data and generalization techniques, our Dyn-
FOA is expected to maintain high-quality audio generation and an
immersive experience in even more complex real-world acoustic
environments.
5
CONCLUSION
We present DynFOA, a first-order spatial audio (FOA) generation
method designed for complex acoustic scenes containing occlu-
sion, reflections, and reverberation. By combining geometrically
and material-aware scene understanding with conditional diffusion
model, DynFOA dynamically adjusts the FOA rendering based on
source motion, ambient acoustics, and listener orientation, ensur-
ing stable and realistic spatial cues even under challenging scenes.
Comprehensive evaluations demonstrate that DynFOA achieves su-
perior FOA quality, clarity, realism, and synchronization. Objec-
tive metrics confirm its high-fidelity modeling of scene acoustics,
while subjective studies demonstrate significant improvements in
occlusion handling, presence, and immersion. Therefore, DynFOA
provides a practical and effective solution for high-fidelity spatial
audio in VR, AR, and immersive media applications.
REFERENCES
[1] Durand R Begault and Leonard J Trejo. 3-d sound for virtual reality
and multimedia. Technical report, 2000. 1
[2] Douglas S Brungart.
Near-field virtual audio displays.
Presence,
11(1):93–106, 2002. 2
[3] Huadai Liu, Tianyi Luo, Kaicheng Luo, Qikai Jiang, Peiwen Sun,
Jialei Wang, Rongjie Huang, Qian Chen, Wen Wang, Xiangtai Li,
et al. Omniaudio: Generating spatial audio from 360-degree video.
In Forty-second International Conference on Machine Learning. 2, 3,
5, 6
[4] Siyi Xie, Hanxin Zhu, Tianyu He, Xin Li, and Zhibo Chen. Sonic4d:
Spatial audio generation for immersive 4d scene exploration. arXiv
preprint arXiv:2506.15759, 2025. 2, 3
[5] Aastha Gupta and Thushara D Abhayapala.
Three-dimensional
sound field reproduction using multiple circular loudspeaker arrays.
IEEE Transactions on Audio, Speech, and Language Processing,
19(5):1149–1159, 2010. 2
[6] William G Gardner and Keith D Martin. Hrtf measurements of a ke-
mar. The Journal of the Acoustical Society of America, 97(6):3907–
3908, 1995. 2
[7] V Ralph Algazi, Richard O Duda, Dennis M Thompson, and Carlos
Avendano. The cipic hrtf database. In Proceedings of the 2001 IEEE
9

<!-- page 10 -->
workshop on the applications of signal processing to audio and acous-
tics (Cat. No. 01TH8575), pages 99–102. IEEE, 2001. 2
[8] Piotr Majdak, Yukio Iwaya, Thibaut Carpentier, Rozenn Nicol,
Matthieu Parmentier, Agnieszka Roginska, Yˆoiti Suzuki, Kankji
Watanabe, Hagen Wierstorf, Harald Ziegelwanger, et al.
Spatially
oriented format for acoustics: A data exchange format representing
head-related transfer functions. In Audio Engineering Society Con-
vention 134. Audio Engineering Society, 2013. 2
[9] Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amen-
gual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, and
Kristen Grauman. Soundspaces: Audio-visual navigation in 3d envi-
ronments. In European conference on computer vision, pages 17–36.
Springer, 2020. 2
[10] Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik,
Alexander Clegg, Paul Calamia, Dhruv Batra, Philip Robinson, and
Kristen Grauman. Soundspaces 2.0: A simulation platform for visual-
acoustic learning. Advances in Neural Information Processing Sys-
tems, 35:8896–8911, 2022. 2
[11] Pedro Morgado, Nuno Nvasconcelos, Timothy Langlois, and Oliver
Wang. Self-supervised generation of spatial audio for 360 video. Ad-
vances in neural information processing systems, 31, 2018. 2
[12] Mert Cokelek, Halit Ozsoy, Nevrez Imamoglu, Cagri Ozcinar, Inci
Ayhan, Erkut Erdem, and Aykut Erdem. Spherical vision transform-
ers for audio-visual saliency prediction in 360-degree videos. IEEE
transactions on pattern analysis and machine intelligence, 2025. 2
[13] Swapnil Bhosale, Haosen Yang, Diptesh Kanojia, Jiankang Deng, and
Xiatian Zhu. Av-gs: Learning material and geometry aware priors
for novel view acoustic synthesis. Advances in Neural Information
Processing Systems, 37:28920–28937, 2024. 2
[14] Xiaomeng Zhang, Hao Sun, Shuopeng Wang, and Jing Xu. A new
regional localization method for indoor sound source based on convo-
lutional neural networks. IEEE Access, 6:72073–72082, 2018. 2
[15] Arda Senocak, Tae-Hyun Oh, Junsik Kim, Ming-Hsuan Yang, and
In So Kweon. Learning to localize sound source in visual scenes. In
Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 4358–4366, 2018. 2
[16] Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson,
Avinatan Hassidim, William T Freeman, and Michael Rubinstein.
Looking to listen at the cocktail party: a speaker-independent audio-
visual model for speech separation. ACM Transactions on Graphics
(TOG), 37(4):1–11, 2018. 2
[17] Zexu Pan, Ruijie Tao, Chenglin Xu, and Haizhou Li. Selective listen-
ing by synchronizing speech with lips. IEEE/ACM Transactions on
Audio, Speech, and Language Processing, 30:1650–1664, 2022. 2
[18] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen,
Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Rit-
ter.
Audio set: An ontology and human-labeled dataset for audio
events. In 2017 IEEE international conference on acoustics, speech
and signal processing (ICASSP), pages 776–780. IEEE, 2017. 2
[19] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman.
Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020
IEEE International Conference on Acoustics, Speech and Signal Pro-
cessing (ICASSP), pages 721–725. IEEE, 2020. 2
[20] Renato Panda.
Multi-modal music emotion recognition: A new
dataset, methodology and comparative analysis. 2
[21] Kranti Kumar Parida, Siddharth Srivastava, and Gaurav Sharma. Be-
yond mono to binaural: Generating binaural audio from mono audio
with depth and cross modal attention. In Proceedings of the IEEE/CVF
winter conference on applications of computer vision, pages 3347–
3356, 2022. 2
[22] Xudong Xu, Hang Zhou, Ziwei Liu, Bo Dai, Xiaogang Wang, and
Dahua Lin. Visually informed binaural audio generation without bin-
aural audios. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 15485–15494, 2021. 2
[23] Francesc Llu´ıs,
Vasileios Chatziioannou,
and Alex Hofmann.
Points2sound: from mono to binaural audio using 3d point cloud
scenes. EURASIP Journal on Audio, Speech, and Music Processing,
2022(1):33, 2022. 3
[24] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad
Norouzi, and William Chan.
Wavegrad: Estimating gradients for
waveform generation. In International Conference on Learning Rep-
resentations. 3
[25] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catan-
zaro. Diffwave: A versatile diffusion model for audio synthesis. In
International Conference on Learning Representations. 3
[26] Felix Kreuk, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexan-
dre D´efossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi.
Audiogen: Textually guided audio generation. In The Eleventh Inter-
national Conference on Learning Representations. 3
[27] Saksham Singh Kushwaha, Jianbo Ma, Mark RP Thomas, Yapeng
Tian, and Avery Bruni. Diff-sage: End-to-end spatial audio generation
using diffusion models. In ICASSP 2025-2025 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP),
pages 1–5. IEEE, 2025. 3, 5, 6
[28] Mojtaba Heydari, Mehrez Souden, Bruno Conejo, and Joshua Atkins.
Immersediffusion: A generative spatial audio latent diffusion model.
In ICASSP 2025-2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025. 3, 5
[29] Zal´an Borsos, Matt Sharifi, Damien Vincent, Eugene Kharitonov, Neil
Zeghidour, and Marco Tagliasacchi. Soundstorm: Efficient parallel
audio generation. 3
[30] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel
Synnaeve, Yossi Adi, and Alexandre D´efossez. Simple and control-
lable music generation. Advances in neural information processing
systems, 36:47704–47720, 2023. 3
[31] Jaeyeon Kim, Heeseung Yun, and Gunhee Kim. Visage: Video-to-
spatial audio generation. In 13th International Conference on Learn-
ing Representations, ICLR 2025, pages 14239–14259. International
Conference on Learning Representations, ICLR, 2025. 3, 6
[32] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya,
Alexander Schwing, and Yuki Mitsufuji. Mmaudio: Taming multi-
modal joint training for high-quality video-to-audio synthesis. In Pro-
ceedings of the Computer Vision and Pattern Recognition Conference,
pages 28901–28911, 2025. 3, 6
[33] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao
Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D
Plumbley. Audioldm 2: Learning holistic audio generation with self-
supervised pretraining. IEEE/ACM Transactions on Audio, Speech,
and Language Processing, 32:2871–2883, 2024. 3
[34] Taha Samavati and Mohsen Soryani. Deep learning-based 3d recon-
struction: a survey. Artificial Intelligence Review, 56(9):9175–9219,
2023. 3
[35] Carl Schissler and Dinesh Manocha. Interactive sound propagation
and rendering for large multi-source scenes. ACM Transactions on
Graphics (TOG), 36(4):1, 2016. 3
[36] Nikunj Raghuvanshi, John Snyder, Ravish Mehra, Ming Lin, and
Naga Govindaraju. Precomputed wave simulation for real-time sound
propagation of dynamic sources in complex scenes. In ACM Siggraph
2010 papers, pages 1–11. 2010. 3
[37] Zhenyu Tang, Hsien-Yu Meng, and Dinesh Manocha. Learning acous-
tic scattering fields for dynamic interactive sound propagation.
In
2021 IEEE Virtual Reality and 3D User Interfaces (VR), pages 835–
844. IEEE, 2021. 3
[38] Bin Lin, Jinlei Zheng, Chaocan Xue, Lei Fu, Ying Li, and Qiang
Shen. Motion-aware correlation filter-based object tracking in satel-
lite videos. IEEE Transactions on Geoscience and Remote Sensing,
62:1–13, 2024. 3
[39] Brian Curless and Marc Levoy. A volumetric method for building
complex models from range images. In Proceedings of the 23rd an-
nual conference on Computer graphics and interactive techniques,
pages 303–312, 1996. 4
[40] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, George
Drettakis, et al. 3d gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4):139–1, 2023. 4
[41] Christophe Van der Kelen, Peter G¨oransson, Bert Pluymers, and Wim
Desmet. On the influence of frequency-dependent elastic properties
in vibro-acoustic modelling of porous materials under structural ex-
citation. Journal of Sound and Vibration, 333(24):6560–6571, 2014.
4
[42] Anton Ratnarajah and Dinesh Manocha.
Listen2scene:
Interac-
10

<!-- page 11 -->
tive material-aware binaural sound propagation for reconstructed 3d
scenes. In 2024 IEEE Conference Virtual Reality and 3D User Inter-
faces (VR), pages 254–264. IEEE, 2024. 4, 5
[43] Lakulish Antani,
Anish Chandak,
Lauri Savioja,
and Dinesh
Manocha.
Interactive sound propagation using compact acoustic
transfer operators. ACM Transactions on Graphics (TOG), 31(1):1–
12, 2012. 4
[44] Logan Courtney and Ramavarapu Sreenivas.
Using deep convolu-
tional lstm networks for learning spatiotemporal features. In Asian
Conference on Pattern Recognition, pages 307–320. Springer, 2019.
4
[45] Franz Zotter and Matthias Frank. Ambisonics: A practical 3D au-
dio theory for recording, studio production, sound reinforcement, and
virtual reality. Springer, 2019. 4, 5
[46] Pedro Morgado, Yi Li, and Nuno Nvasconcelos. Learning representa-
tions from audio-visual spatial alignment. Advances in Neural Infor-
mation Processing Systems, 33:4733–4744, 2020. 4, 5
[47] Boaz Rafaely. Fundamentals of spherical array processing, volume 8.
Springer, 2015. 5
[48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser,
and Bj¨orn Ommer. High-resolution image synthesis with latent diffu-
sion models. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 10684–10695, 2022. 5
[49] Zhizhang Chen, Gopal Gokeda, and Yiqiang Yu.
Introduction to
Direction-of-arrival Estimation. Artech House, 2010. 6
[50] Philipos C Loizou. Speech enhancement: theory and practice. CRC
press, 2007. 6
[51] Salvador Cerd´a, Alicia Gim´enez, Jinson Romero, Rosa Cibrian, and
JL Miralles. Room acoustical parameters: A factor analysis approach.
Applied Acoustics, 70(1):97–109, 2009. 6
[52] Ryuichi Yamamoto, Eunwoo Song, and Jae-Min Kim. Parallel wave-
gan: A fast waveform generation model based on generative adver-
sarial networks with multi-resolution spectrogram. In ICASSP 2020-
2020 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 6199–6203. IEEE, 2020. 6
[53] Simon Dahl Jepsen, Mads Græsbøll Christensen, and Jesper Rindom
Jensen.
A study of the scale invariant signal to distortion ra-
tio in speech separation with noisy references.
arXiv preprint
arXiv:2508.14623, 2025. 6
11
