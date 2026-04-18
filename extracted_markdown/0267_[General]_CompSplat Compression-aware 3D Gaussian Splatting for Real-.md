# CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video

Hojun Songâ1 Heejung Choiâ1 Aro Kim1 Chae-yeong Song1 Gahyeon Kim1 Soo Ye Kimâ 2 Jaehyup Leeâ 1 Sang-hyo Parkâ 1

1Kyungpook National University 2Adobe Research

https://hdddhdd.github.io/CompSplat-page/

(c) NoPe-NeRF (CVPR 2024)  
<!-- image-->  
(d) LocalRF (CVPR 2024)  
(e) LongSplat (ICCV 2025)  
(g) GT  
(f) Ours  
Figure 1. CompSplat achieves high-quality novel view synthesis from real-world compressed videos. Given (a) compressed video input, our approach leverages (b) compression information showing per-frame quality variations from different quantization parameters. Due to degraded inputs from compression, previous methods (c) NoPe-NeRF, (d) LocalRF, and (e) LongSplat generate blurry or distorted results. In contrast, through compression-aware optimization, (f) our proposed method produces clear reconstructions with fine details.

## Abstract

High-quality novel view synthesis (NVS) from real-world videos is crucial for applications such as cultural heritage preservation, digital twins, and immersive media. However, real-world videos typically contain long sequences with irregular camera trajectories and unknown poses, leading to pose drift, feature misalignment, and geometric distortion during reconstruction. Moreover, lossy compression amplifies these issues by introducing inconsistencies that gradually degrade geometry and rendering quality. While recent studies have addressed either long-sequence NVS or unposed reconstruction, compression-aware approaches still focus on specific artifacts or limited scenarios, leaving diverse compression patterns in long videos insufficiently explored. In this paper, we propose Comp-Splat, a compression-aware training framework that explic-

itly models frame-wise compression characteristics to mitigate inter-frame inconsistency and accumulated geometric errors. CompSplat incorporates compression-aware frame weighting and an adaptive pruning strategy to enhance robustness and geometric consistency, particularly under heavy compression. Extensive experiments on challenging benchmarks, including Tanks and Temples, Free, and Hike, demonstrate that CompSplat achieves state-of-the-art rendering quality and pose accuracy, significantly surpassing most recent state-of-the-art NVS approaches under severe compression conditions.

## 1. Introduction

High-quality 3D reconstruction and novel view synthesis (NVS) have become central to applications such as virtual and augmented reality, digital twins, and immersive media production [14, 27]. By reconstructing reliable threedimensional structure from captured images and videos, these technologies enable spatial understanding of real scenes and natural humanâcomputer interaction. With the rapid growth of video content on platforms like YouTube and the rising demand for real-world 3D experiences, it is becoming increasingly important to move beyond carefully curated datasets and develop robust methods that remain reliable when operating directly on videos captured in uncontrolled real-world environments [24, 30, 36, 48].

Unlike benchmark datasets commonly used in the NVS literature, real-world videos exhibit far more challenging characteristics. Such videos often span hundreds or thousands of frames, contain irregular camera trajectories, and suffer from lighting variations that destabilize pose estimation and geometry recovery [19, 33]. More importantly, nearly all real-world videos undergo lossy compression, during capture on consumer devices or after upload to online platforms, using codecs such as JPEG [35], H.264/AVC [40], or HEVC [31]. Compression artifacts from these codecs degrade spatial detail, disrupt temporal coherence, and severely impair downstream processes including keypoint matching, pose estimation, and geometric refinement.

Existing approaches, which are typically designed and validated on clean and short sequences, struggle to maintain geometric stability or rendering quality when faced with the combined challenges of long-range inconsistency and heterogeneous compression artifacts. The mismatch between controlled research datasets and real-world inputs significantly undermines the practicality of NVS pipelines. Consequently, achieving robust reconstruction from long, compressed video remains as essential yet underexplored challenge. As shown in Fig. 1, NoPe-NeRF [4], LocalRF [26], and LongSplat [19] produce structure collapse and geometric instability when frames are compressed under typical real-world settings, highlighting the necessity of modeling compression information during reconstruction.

To address such challenges, we present CompSplat, an adaptive optimization framework tailored for long video sequences under real compression scenarios. CompSplat estimates a frame-wise reliability factor by jointly modeling compression indicators (e.g., quantization parameters, bitrates) and training stability cues including pose estimation confidence and keypoint matching robustness. These weighting factors are continuously updated during training and directly regulate the reconstruction dynamics: frames with high reliability receive denser Gaussian updates, whereas low-quality frames are down-weighted or pruned to prevent the accumulation of compression-induced error and to avoid propagating unstable gradients caused by severely degraded frames. This compression-aware adaptive strategy substantially stabilizes optimization, mitigates geometric distortions, and preserves cross-frame consistency, even in severely compressed or bandwidth-

constrained scenarios.

Extensive experiments on compressed versions of longvideo datasets, including Free [37] and Tanks and Temples [15] demonstrate that CompSplat consistently outperforms recent state-of-the-art NVS baselines by large margin. In particular, compared to standard 3DGS methods that do not consider compression, CompSplat significantly improves rendering fidelity, pose accuracy, and overall training stability across diverse compression levels and longsequence settings. By explicitly modeling real-world degradations, CompSplat meaningfully enhances the practical usability and robustness of Gaussian Splatting-based reconstruction. Our key contributions are summarized as follows:

â¢ We propose a compression-aware GS framework (Comp-Splat) for long compressed real-world videos.

â¢ We firstly design an adaptive frame-wise reliability score that integrates compression metrics and stability cues to dynamically guide Gaussian densification and pruning.

â¢ We demonstrate that our CompSplat achieves robust and high-fidelity NVS under practical compression scenarios through comprehensive evaluations across challenging long-sequence GS benchmarks.

## 2. Related Work

Novel View Synthesis. Novel view synthesis is a fundamental task in 3D scene understanding that generates images from new viewpoints given a limited number of input images. Neural Radiance Fields (NeRF) [27] achieved photorealistic results through implicit neural representations, but it suffers from slow rendering speeds and long optimization times due to volumetric rendering. To overcome these limitations, 3D Gaussian Splatting (3DGS) [14] was proposed. 3DGS represents scenes with millions of Gaussian primitives and enables real-time rendering via differentiable rasterization while maintaining NeRF-level quality. Subsequently, 3DGS research has been conducted in various directions, such as reducing pose estimation dependency [8, 13], improving the handling of long videos and large-scale scenes [19, 26], and addressing dynamic scene reconstruction [41]. However, most methods were evaluated on high-quality datasets like Mip-NeRF 360 [3] and Tanks and Temples [15] with uncompressed images, and their robustness on long compressed real-world videos remains underexplored.

3D Reconstruction from Real-World Videos. Real-world videos often contain long sequences, irregular camera motion, and incomplete pose information, leading to pose drift, unstable feature matching, and temporal inconsistency in 3D reconstruction. Various approaches have been proposed to address these challenges. LocalRF [26] adopts progressive optimization, VastGaussian [20] partitions large scenes, Scaffold-GS [23] introduces anchor-based Gaussian representations, and LongSplat [19] achieves robust unposed video reconstruction through incremental joint optimization. Traditional methods like COLMAP [29] and SLAM systems [5, 34] explicitly estimate poses, while neural rendering methods such as BARF [18], NoPe-NeRF [4], and CF-3DGS [8] jointly learn geometry and poses. However, existing methods largely overlook another critical characteristic of real-world videos: lossy compression. Most videos undergo compression during smartphone storage or online uploads, yet the impact of compression artifacts on 3D reconstruction remains underexplored.

<!-- image-->  
Figure 2. Overview of the CompSplat pipeline: (a) Our approach builds upon an unposed-GS framework, reconstructing a 3D Gaussian scene from compressed videos through incremental pose estimation and optimization. (b) Frame-wise compression information (QP and bitrates) is converted into a confidence score. (c) We introduce Quality-guided Density Control, which regulates Gaussian optimization based on frame reliability: (c.1) Scale-based pruning removes over-diffused Gaussians that primarily arise in low-quality frames by leveraging frame confidence. (c.2) Adaptive Densification and Pruning adjust densification gradient and pruning opacity thresholds based on frame confidence. (d) Quality Gap-aware Masking mitigates frame-to-frame quality differences by applying a gap ratioâbased pixel mask.

Video Compression in 3D Vision Tasks. Most videos captured or distributed in the real world are compressed in a lossy manner due to storage and transmission constraints, inevitably introducing artifacts. Furthermore, encoder design including different frame types and QP variation among frames induce per-frame quality fluctuations across a sequence. Consequently, in 2D vision, a large body of compression-aware restoration methods addresses these issues by explicitly accounting for compression artifacts or exploiting multi-frame cues to counter quality oscillations [1, 7, 10, 17, 38, 39, 44â47, 51]. In particular, many studies recently utilize coding priors to adaptively control the gain of each channel and to adjust restoration weights according to compression characteristics [6, 22, 43, 49, 50, 54]. In 3D vision, however, very few studies have examined the impact of video compression on reconstruction quality or explicitly modeled frame-level quality variations caused by codec design. This gap is particularly evident in NVS. In the NeRF domain, recent efforts have explored robustness to degraded or noisy inputs [12, 28, 36, 52, 53], but these typically focus on improving overall input image quality rather than addressing compression-induced artifacts or frame-level quality variations that directly affect the 3D reconstruction process. In the context of 3DGS, HQGS [21] enhances 3DGS robustness to degradations such as blur and coding artifact, but it does not address temporal inconsistencies or framelevel quality fluctuations, which is inherently prone to appear in compressed videos. As a result, its applicability to long, compressed video inputs is limited.

Position of Our Work. Existing NVS and 3DGS-based reconstruction methods mainly assume clean inputs and do not explicitly account for codec-induced frame quality variations, while compression-aware approaches in 2D vision focus on image restoration rather than stabilizing longsequence 3D optimization. Our work bridges these directions by introducing a compression-aware framework for long-video NVS that operates directly on compressed and unposed real-world videos and explicitly leverages codec information to model frame-wise reliability. In particular, our CompSplat complements prior unposed 3DGS methods such as LongSplat [19] and CF-3DGS [8] by integrating compression indicators and training stability cues into the optimization process, enabling robust geometry and pose reconstruction under practical real-world compression conditions.

## 3. Preliminaries

3D Gaussian Splatting. In 3D Gaussian Splatting [14], a scene is represented as a set of anisotropic 3D Gaussians, each parameterized by a mean $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , a covariance matrix $\bar { \mathbf { \Sigma } } = \mathbf { R S S } ^ { \top } \mathbf { R } ^ { \top }$ , and appearance attributes such as color and opacity Î±. During rendering, each Gaussian is projected onto the image plane through the camera pose W, and the final pixel value is obtained by alpha blending along the ray:

$$
C = \sum _ { i } c _ { i } \alpha _ { i } \prod _ { j < i } ( 1 - \alpha _ { j } ) .\tag{1}
$$

Anchor-based Representation. Building on the anchorbased Scaffold-GS [23], LongSplat [19] associates each anchor v at position $\mathbf { x } _ { v }$ with k Gaussians whose centers are defined by relative offsets {Oi} and a scale factor $l _ { v }$ :

$$
\begin{array} { r } { \pmb { \mu } _ { i } = \mathbf { x } _ { v } + \mathbf { O } _ { i } \cdot l _ { v } . } \end{array}\tag{2}
$$

Gaussian attributes are predicted from anchor features using lightweight MLP heads.

Optimization Pipeline. LongSplat [19] reconstructs unposed long videos through an incremental optimization pipeline composed of three stages: (a) Pose Estimation, which aligns each frame via correspondence-guided PnP and photometric refinement; (b) Local Optimization, which updates visible Gaussians within a short temporal window to preserve local consistency; and (c) Global Refinement, which periodically optimizes all Gaussians and poses for sequence-wide coherence. The overall objective combines photometric, depth, and reprojection losses:

$$
L _ { \mathrm { t o t a l } } = L _ { \mathrm { p h o t o } } + L _ { \mathrm { d e p t h } } + L _ { \mathrm { r e p r o j e c t i o n } } .\tag{3}
$$

Gaussian densification and pruning are further applied to regulate primitive density throughout training.

## 4. Method

We propose a compression-aware pipeline for long-video novel view synthesis that adapts the 3D Gaussian optimization process to frame-wise quality variations inherent in compressed videos. As shown in Fig. 2, the pipeline consists of two components: (i) Quality-guided Density Control, which modulates Gaussian densification and pruning based on codec-derived frame reliability, and (ii) Quality Gap-aware Masking, which mitigates view-level quality disparities by applying keypoint-driven pixel masking to the photometric supervision. Together, these components mitigate quality imbalance across frames and yield stable, highfidelity reconstruction under practical compression settings.

## 4.1. Problem Definition and Objectives

Problem Definition. Video compression removes interframe redundancy under bitrate constraints through intraand inter-prediction. As illustrated in Fig. 3, these codecdriven variations induce substantial fluctuations in bit allocation and PSNR across a sequence, leading to inconsistent preservation of high-frequency details. Such frame-wise quality inconsistency leads to two major issues in existing unposed-3DGS frameworks. First, density optimization becomes unreliable: fixed densification and pruning thresholds treat all frames equally, causing high-quality frames to be over-pruned and low-quality frames to introduce noisy Gaussians. Second, quality gaps between adjacent frames degrade feature matching and pose estimation reliability, yielding unstable view supervision and inconsistent updates during incremental optimization.

<!-- image-->  
Figure 3. Frame-wise compression analysis. During video compression, each frame is encoded with QP values, leading to substantial inter-frame variations in PSNR and bitrate. This nonuniformity becomes more pronounced in long real-world videos, highlighting the need to explicitly consider frame-wise compression artifacts when applying 3DGS to compressed video.

Objectives. To address these two issues, we develop a compression-aware pipeline that adapts Gaussian optimization process to frame-level compression characteristics: (i) We regulate Gaussian densification and pruning using a frame confidence score, enabling density control that responds to quality variations across frames; and (ii) We mitigate quality gaps by modulating photometric supervision with a view-level gap score, reducing the influence of unreliable frames. Together, these objectives ensure consistent density evolution and stable view supervision in compressed long-video novel view synthesis.

## 4.2. Quality-guided Density Control

Adaptive Frame Quality Estimation. Mordern video codecs can assign different bits and QP to each frame at the cost of quality fluctuation per frame. Obviously, frames with lower QP values retain more high-frequency details, whereas frames with higher QP values tend to lose fine structures. Similarly, frames with higher bitrates means that more information is needed due to increased scene complexity. To quantify these characteristics, we newly introduce a QP-based confidence score $q _ { t } ^ { q }$ and a bitrate-based confidence score $q _ { t } ^ { b } ,$ , and then we combine them into a unified frame confidence score $q _ { t } = q _ { t } ^ { q } + q _ { t } ^ { b }$ . Both confidence terms are computed through linear normalization of the respective value over the sequence:

$$
q _ { t } ^ { q } = \lambda ^ { q } \frac { Q _ { \operatorname* { m a x } } ^ { f } - Q _ { t } ^ { f } } { Q _ { \operatorname* { m a x } } ^ { f } - Q _ { \operatorname* { m i n } } ^ { f } + \varepsilon } , \quad q _ { t } ^ { b } = \lambda ^ { b } \frac { B _ { t } ^ { f } - B _ { \operatorname* { m i n } } ^ { f } } { B _ { \operatorname* { m a x } } ^ { f } - B _ { \operatorname* { m i n } } ^ { f } + \varepsilon } .\tag{4}
$$

where $Q _ { t } ^ { f }$ and $B _ { t } ^ { f }$ denote the QP and the bits at the corresponding frame t, respectively. Îµ is a small constant to avoid division by zero, and Î»q and $\lambda ^ { b }$ are weighting hyperparameters controlling the QP and bitrate contributions, of frame $t ,$ respectively. $\mathbf { \bar { Q } } _ { \operatorname* { m i n } } ^ { f } , \mathbf { \bar { Q } } _ { \operatorname* { m a x } } ^ { f } , B _ { \operatorname* { m i n } } ^ { f } ,$ and $B _ { \mathrm { m a x } } ^ { f }$ represent the minimum and maximum values across the sequence. To ensure temporal stability across frames, an exponential moving average (EMA) is applied:

$$
\bar { q } _ { t } = \beta \bar { q } _ { t - 1 } + ( 1 - \beta ) q _ { t } ,\tag{5}
$$

where $\beta$ denotes the momentum parameter. The resulting EMA confidence $\bar { q } _ { t }$ serves as a baseline for scale-based pruning and adaptive Gaussian density control in subsequent stages.

Gaussian Scale-based Pruning. In GS optimization, the spatial density of primitives is determined by two criteria: (i) new Gaussians are densified when their accumulated gradient exceeds a threshold Î¸, and (ii) existing ones are pruned when their opacity Î± falls below a threshold Ï:

$$
\mathrm { d e n s i f y ~ i f } \ : g _ { t } > \theta \quad \mathrm { a n d } \quad \mathrm { p r u n e ~ i f } \ : \alpha _ { t } < \omega ,\tag{6}
$$

where $g _ { t }$ and $\alpha _ { t }$ denote the gradient magnitude and opacity of a Gaussian at frame $t ,$ respectively. In our pipline, scalebased pruning modifies only the pruning rule by making Ï responsive to Gaussian scale.

Compressed frames often lose high-frequency components such as fine textures and edges, producing blurred regions that generate over-diffused Gaussians. To filter out these unreliable primitives, we introduce a scale-based pruning step. For each anchor v, we compute a representative scale by averaging its Gaussian offsets and normalizing it by the median scale in the scene:

$$
U _ { v } = \| \mathbf { s } ^ { ( v ) } \| _ { 2 } , \qquad { \tilde { U } } _ { v } = \frac { U _ { v } } { \mathrm { m e d i a n } ( U ) } .\tag{7}
$$

Large normalized scales $\tilde { U } _ { v }$ indicate that the Gaussians are likely originating from degraded frames. To couple scale

filtering with frame reliability, we adaptively modulate the pruning threshold per frame as:

$$
\omega _ { t } = \omega _ { 0 } \exp ( \bar { q } _ { t } \tilde { U } _ { v } ) ,\tag{8}
$$

where $\omega _ { 0 }$ is the base opacity threshold and ${ \bar { q } } _ { t }$ is the EMA confidence from Eq. 5. This formulation enables frameadaptive removal of over-scaled Gaussians while preserving compact, reliable primitives.

Adaptive Densification and Pruning. Once over-diffused Gaussians are removed through scale-based pruning, reliable primitives receive clearer gradient signals, allowing density evolution to respond more faithfully to frame-wise quality variations. We therefore adaptively modulate the thresholds Î¸ and Ï using frame confidence to stabilize densification and pruning under varying compression levels. Given the current confidence $q _ { t }$ and its temporal average ${ \bar { q } } _ { t }$ , we define:

$$
\theta _ { t } = \theta _ { 0 } \exp ( \bar { q } _ { t } - q _ { t } ) , \qquad \omega _ { t } ^ { \prime } = \omega _ { 0 } \exp ( \bar { q } _ { t } - q _ { t } ) ,\tag{9}
$$

where $\theta _ { 0 }$ and $\omega _ { 0 }$ are base thresholds. When $q _ { t } > \bar { q } _ { t }$ , the model reduces $\theta _ { t }$ to allow denser Gaussian creation and decreases $\omega _ { t } ^ { \prime }$ to relax pruning. Conversely, $q _ { t } < \bar { q } _ { t }$ increases both thresholds, suppressing Gaussian creation and enforcing stronger pruning. This confidence-driven modulation of Î¸ and Ï stabilizes density evolution and prevents inconsistent Gaussian growth across frames with varying compression quality.

## 4.3. Quality Gap-aware Masking

Frame-level compression variability creates quality gaps (Fig. 3) that adversely affect view supervision, corresponding to the second objective described in Sec. 4.1. In LongSplat [19], each incoming frame is aligned through MASt3R[16]-based keypoint matching followed by PnP initialization, after which newly observed regions are unprojected to expand the Gaussian set. However, when frames suffer from compression artifacts or reduced detail, the number of detected keypoints decreases, resulting in a less number of geometrically consistent correspondences. This degradation reduces pose reliability and destabilizes the subsequent visibility-adapted local optimization.

To address this quality gap, we compute an inlier ratio for each view t using the number of detected keypoints $K _ { t }$ and the number of inlier correspondences $I _ { t }$ returned by MASt3R [16]:

$$
r _ { t } = \frac { I _ { t } } { K _ { t } + \varepsilon } ,\tag{10}
$$

where $\varepsilon$ is a small constant to prevent division by zero. This ratio serves as a view-level reliability measure: high-quality frames generally produce more consistent matches, whereas degraded frames yield low inlier ratios due to loss of structure from compression.

Table 1. Quantitative comparison on the Free dataset [37] across various baseline methods. âOOMâ indicates an out-of-memory failure. âUncomp.â refers to training and evaluation on original uncompressed videos. Our method consistently achieves higher PSNR, SSIM, and LPIPS across all scenes by explicitly accounting for compressed video artifacts, leading to more stable and accurate reconstructions under challenging camera trajectories and complex geometric structures.
<table><tr><td rowspan="2">Scenes</td><td colspan="3">LongSplat (Uncomp.)</td><td colspan="3">NoPe-NeRF [4]</td><td colspan="3">LocalRF [26]</td><td colspan="3">CF-3DGS [8]</td><td colspan="3">CF-3DGS + Ours</td><td colspan="3">LongSplat [19]</td><td colspan="3">LongSplat + Ours</td></tr><tr><td>| PSNRâ SSIMâ LPIPSâ|</td><td></td><td></td><td>| PSNRâ SSIMâ LPIPSâ|</td><td></td><td></td><td></td><td></td><td>| PSNRâ SSIMâ LPIPSâ|</td><td>| PSNRâ</td><td></td><td>SSIMâ LPIPSâ</td><td></td><td></td><td>| PSNRâ SSIMâ LPIPSâ|</td><td></td><td>| PSNRâ SSIMâ LPIPSâ|</td><td></td><td></td><td>|PSNRâ SSIMâ LPIPSâ</td><td></td></tr><tr><td>Grass</td><td>26.03</td><td>0.79</td><td>0.21</td><td>17.84</td><td>0.41</td><td>0.73</td><td>20.20</td><td>0.47</td><td>0.44</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>23.61</td><td>0.64</td><td>0.33</td><td>24.13</td><td>0.66</td><td>0.30</td></tr><tr><td>Hydrant</td><td>23.23</td><td>0.73</td><td>0.21</td><td>19.43</td><td>0.55</td><td>0.59</td><td>20.34</td><td>0.59</td><td>0.31</td><td>12.63</td><td>0.33</td><td>0.61</td><td>9.19</td><td>0.19</td><td>0.66</td><td>23.55</td><td>0.72</td><td>0.27</td><td>24.28</td><td>0.67</td><td>0.30</td></tr><tr><td>Lab</td><td>26.36</td><td>0.86</td><td>0.16</td><td>17.61</td><td>0.55</td><td>0.61</td><td>17.25</td><td>0.55</td><td>0.31</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>26.38</td><td>0.86</td><td>0.18</td><td>26.87</td><td>0.86</td><td>0.17</td></tr><tr><td>Pillar</td><td>28.83</td><td>0.84</td><td>0.19</td><td>19.07</td><td>0.59</td><td>0.62</td><td>25.74</td><td>0.70</td><td>0.34</td><td>13.92</td><td>0.41</td><td>0.66</td><td>14.78</td><td>0.39</td><td>0.64</td><td>26.15</td><td>0.71</td><td>0.34</td><td>26.96</td><td>0.74</td><td>0.32</td></tr><tr><td>Road</td><td>21.54</td><td>0.60</td><td>0.34</td><td>19.73</td><td>0.63</td><td>0.62</td><td>22.54</td><td>0.64</td><td>0.38</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>20.91</td><td>0.59</td><td>0.43</td><td>21.75</td><td>0.62</td><td>0.41</td></tr><tr><td>Sky</td><td>26.67</td><td>0.86</td><td>0.19</td><td>16.08</td><td>0.59</td><td>0.58</td><td>19.56</td><td>0.65</td><td>0.28</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>25.28</td><td>0.82</td><td>0.27</td><td>25.77</td><td>0.83</td><td>0.26</td></tr><tr><td>Stair</td><td>29.21</td><td>0.84</td><td>0.19</td><td>20.55</td><td>0.62</td><td>0.59</td><td>26.08</td><td>0.81</td><td>0.20</td><td>14.32</td><td>0.43</td><td>0.60</td><td>12.01</td><td>0.34</td><td>0.63</td><td>26.97</td><td>0.77</td><td>0.29</td><td>27.41</td><td>0.78</td><td>0.28</td></tr><tr><td>Avg</td><td>25.98</td><td>0.79</td><td>0.21</td><td>18.62</td><td>0.56</td><td>0.62</td><td>21.96</td><td>0.63</td><td>0.32</td><td>13.62</td><td>0.39</td><td>0.62</td><td>12.00</td><td>0.31</td><td>0.65</td><td>24.69</td><td>0.73</td><td>0.30</td><td>25.31</td><td>0.74</td><td>0.29</td></tr></table>

Table 2. Quantitative evaluation of camera pose estimation accuracy on the Free dataset [37].
<table><tr><td>Method</td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>LongSplat (Uncomp.)</td><td>0.616</td><td>1.447</td><td>0.010</td></tr><tr><td>NoPe-NeRF</td><td>5.834</td><td>4.716</td><td>0.545</td></tr><tr><td>CF-3DGS</td><td>0.807</td><td>6.241</td><td>0.030</td></tr><tr><td>CF-3DGS + Ours</td><td>0.824</td><td>6.709</td><td>0.029</td></tr><tr><td>LongSplat</td><td>0.872</td><td>1.721</td><td>0.016</td></tr><tr><td>LongSplat + Ours</td><td>0.539</td><td>1.047</td><td>0.008</td></tr></table>

To reduce the influence of these unreliable views, we convert the inlier ratio into a pixel drop rate:

$$
d _ { t } = \eta ( 1 - r _ { t } ) ,\tag{11}
$$

where Î· controls the sensitivity to quality gaps. Then, a Bernoulli mask with probability $d _ { t }$ is applied to the photometric loss, discarding a proportion of pixels in accordance with frame reliability. Consequently, reliable views provide stronger supervision, while less reliable views contribute less, yielding stable view-level optimization despite large frame-to-frame quality differences.

## 5. Experiments

## 5.1. Experimental Setup

Compression Settings. To reflect real-world encoding environments as in [2], all videos were encoded using x265 codec [42] at 60 fps with YUV 4:2:0 color sampling. A GOP length of 32 and a Random Access coding structure were adopted, while Open-GOP was disabled. We used a compression level corresponding to QP 37 as common practice to evaluate the quality [11], and more QP-variants can be found at supplementary material.

Datasets. Using the compression configuration described, we evaluate our pipeline on three real-world video datasets.

â¢ Tanks and Temples [15]: Eight forward-facing scenes; trained at full resolution and evaluated on every 9th frame.

Table 3. Quantitative evaluation of novel view synthesis quality on the Tanks and Temples dataset [15]. Bold indicates the best within each baseline +Ours pair
<table><tr><td>Method</td><td>PSNRâ SSIMâ</td><td>LPIPSâ</td><td></td><td>RPEt â</td><td>RPErâ</td><td>ATEâ</td></tr><tr><td>CF-3DGS</td><td>27.36</td><td>0.80</td><td>0.27</td><td>0.052</td><td>0.099</td><td>0.006</td></tr><tr><td>CF-3DGS + Ours</td><td>27.84</td><td>0.82</td><td>0.27</td><td>0.049</td><td>0.085</td><td>0.005</td></tr><tr><td>LongSplat</td><td>28.21</td><td>0.82</td><td>0.26</td><td>2.265</td><td>0.831</td><td>0.031</td></tr><tr><td>LongSplat + Ours</td><td>28.37</td><td>0.82</td><td>0.26</td><td>2.176</td><td>0.823</td><td>0.029</td></tr></table>

Table 4. Quantitative evaluation on the Hike dataset. Comparison between LongSplat and our method in terms of photometric quality and pose accuracy.
<table><tr><td>Method</td><td>| PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEr â ATEâ</td><td></td></tr><tr><td>LongSplat </td><td>19.37</td><td>0.52</td><td>0.36</td><td>1.180</td><td>6.639</td><td>0.025</td></tr><tr><td>Ours</td><td>19.65</td><td>0.54</td><td>0.35</td><td>1.152</td><td>6.185</td><td>0.024</td></tr></table>

â¢ Free Dataset [37]: Seven handheld videos with unconstrained camera motion; trained at half resolution and evaluated on every 9th frame.

â¢ Hike Dataset [25]: Long outdoor videos with complex motion and large-scale geometry; trained at 1/4 resolution with half the frames, evaluated every 10th frame.

Baselines. We compare our method with unposed reconstruction approaches, including NoPe-NeRF [4], LocalRF [26], CF-3DGS [8], and LongSplat [19].

Implementation Details. We adopt LongSplat [19] and CF-3DGS [8] as our baseline framework and integrate our proposed method, CompSplat, within the pipeline. Following the original LongSplat configuration, the number of iterations for the local, global, and refine stages is set to 400, 900, and 10,000, respectively. All experiments are conducted on an NVIDIA 3080 GPU, while CF-3DGS is trained on an NVIDIA A6000 GPU. For our method, the confidence weighting parameters are set to Î»q = 1.0 and $\lambda ^ { b } = 0 . 5$ , the drop-rate scaling parameter to Î· = 0.5, and the numerical stability constant $\mathbf { t o } \varepsilon = 1 0 ^ { - 6 }$

<!-- image-->  
NoPe-NeRF

<!-- image-->  
LocalRF

<!-- image-->  
CF-3DGS

<!-- image-->  
LongSplat

<!-- image-->  
Ours

<!-- image-->  
GT

Figure 4. Qualitative comparison on the Free dataset [37]. We compare our method against CF-3DGS [8], NoPe-NeRF [4], LocalRF [26], and LongSplat [19]. CF-3DGS produces highly diffused or incorrect Gaussians on compressed datasets, making object and scene structures difficult to recognize. Other baseline methods also yield blurry or geometrically distorted reconstructions. LongSplat performs relatively well; however, when combined with our approach, the results exhibit sharper textures and clearer object boundaries, demonstrating improved reconstruction quality even under compressed conditions.  
<!-- image-->  
Figure 5. Visualization of camera trajectories on the Free dataset [37]. CF-3DGS fails to estimate a valid camera trajectory due to OOM when processing long sequences. Compared to LongSplat and other baselines, our method produces more stable and consistent camera trajectories under compressed video frames.

## 5.2. Comparisons

Free Dataset. Tabs. 1 and 2 show that our method achieves state-of-the-art performance in both rendering quality and camera pose estimation. Figs. 4 and 5 demonstrate that our method better preserves fine textures and object boundaries and provides more accurate camera trajectory estimation, resulting in clearer and more stable reconstructions than existing baselines.

Tank and Temples Dataset. Tab. 3 presents the quantitative results on the Tanks and Temples dataset [15]. Across all evaluated baselines, adding our method consistently improves the results, indicating that our approach generalizes well and provides stable gains on this benchmark. Fig. 6 further shows that our method preserves fine geometric details and maintains texture consistency in challenging regions such as reflective surfaces and thin structural elements.

<!-- image-->  
CF-3DGS

<!-- image-->  
LongSplat

<!-- image-->  
LongSplat + Ours

<!-- image-->  
GT  
Figure 6. Qualitative results on Tank and Temples dataset [15]. Comparison of rendered views across various method with ours.

Hike Dataset. Tab. 4 compares LongSplat [19] and our method on the Hike dataset [25]. Our method consistently outperforms the LongSplat baseline across all photometric and pose metrics, demonstrating robust performance even in complex scenes. Please refer to the Supplementary Material for the full per-scene quantitative results.

## 5.3. Ablation Study

Ablation on optimization components. Tab. 5 reports an ablation study of three components: (a) Gaussian Scalebased Pruning, (b) Adaptive Densification and Pruning, and (c) Quality Gap-aware Masking. When each component is applied individually, the performance improves to varying degrees in both photometric and camera metrics. Combining two components yields comparable or slightly improved results, while using all components (Ours) achieves the best performance across all metrics.

Table 5. Ablation on Compression-aware optimization components. Effects of (a) Gaussian Scale-based Pruning, (b) Adaptive Densification and Pruning, and (c) Quality Gap-aware Masking, averaged over all Free dataset [37] scenes.
<table><tr><td colspan="2">Method</td><td rowspan="2">PSNRâ</td><td rowspan="2">SSIMâ</td><td rowspan="2">LPIPSâ</td><td rowspan="2">RPEtâ</td><td rowspan="2">RPErâ</td><td rowspan="2">ATEâ</td></tr><tr><td>(a) (b)</td><td>()</td></tr><tr><td rowspan="4"> $\checkmark$ </td><td rowspan="4">â â</td><td>24.69</td><td>0.73</td><td>0.30</td><td>0.872</td><td>1.721</td><td>0.016</td></tr><tr><td>24.96</td><td>0.74</td><td>0.29</td><td>0.680</td><td>1.566</td><td>0.012</td></tr><tr><td>24.91</td><td>0.73</td><td>0.29</td><td>0.716</td><td>1.171</td><td>0.012</td></tr><tr><td>24.92</td><td>0.74</td><td>0.30</td><td>0.782</td><td>1.513</td><td>0.011</td></tr><tr><td> $\checkmark$  â</td><td>â</td><td></td><td>24.83</td><td>0.73</td><td>0.30</td><td>0.759</td><td>1.580</td><td>0.013</td></tr><tr><td rowspan="2"></td><td></td><td>â</td><td>24.73</td><td>0.73</td><td>0.30</td><td>0.708</td><td>1.707</td><td>0.012</td></tr><tr><td>â</td><td>â</td><td>24.72</td><td>0.73</td><td>0.30</td><td>0.802</td><td>1.405</td><td>0.012</td></tr><tr><td>â</td><td>â</td><td>â</td><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr></table>

Table 6. Ablation of frame-quality estimation components on the Free dataset [37]. QP-only and bitrate-only scoring both reduce accuracy, while the combined formulation yields the best photometric and pose results.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>Bit-only</td><td>24.81</td><td>0.73</td><td>0.29</td><td>0.882</td><td>2.086</td><td>0.015</td></tr><tr><td>QP-only</td><td>21.92</td><td>0.62</td><td>0.36</td><td>1.179</td><td>4.111</td><td>0.017</td></tr><tr><td>Together (Ours)</td><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr></table>

Ablation on Frame-Quality Estimation. Tab. 6 analyzes frame-quality confidence based on QP only or bitrate only. Both single-factor designs degrade performance on photometric and camera metrics, with QP-only being particularly limited due to its coarse, discrete nature. In contrast, combining QP and bitrate yields the best overall results.

Video Codec Comparison. Tab. 7 evaluates performance across widely used video codecs from different generations, including VVC, H.264, and H.265. Our method consistently outperforms LongSplat across all codecs, indicating stable performance under different compression standards.

Real-World Compressed Video Evaluation. Fig. 7 and Tab. 8 present qualitative and quantitative comparisons on real-world videos captured using a mobile device and compressed under our setting. Our method outperforms LongSplat across all metrics. Qualitatively, our results better preserve textures and edge details under compression, as evidenced by the error heatmaps in Fig. 7, where the rendered results of our method exhibit fewer warm-colored regions, indicating smaller deviations from the ground truth.

## 6. Conclusion

We presented CompSplat, a compression-aware 3DGS framework for real-world long and unposed video reconstruction. By explicitly modeling frame-wise compression characteristics, CompSplat adaptively regulates Gaussian densification and pruning, mitigating geometric drift and maintaining consistent reconstruction quality under varying compression levels. Extensive experiments demonstrate that CompSplat achieves state-of-the-art performance on real compressed videos, outperforming existing unposed 3DGS methods in both quality and stability. These results highlight the effectiveness of our approach in practical scenarios, advancing the robustness and generalization of 3DGS for real-world, practical bandwidth-constrained video applications. We believe this work lays an important foundation for compressionaware 3DGS in long unposed video reconstruction.

Table 7. Comparison across different video codecs. Evaluation on the Free dataset [37] under VVC, H.264, and H.265 (QP37) compression, comparing LongSplat and our method.
<table><tr><td>Method</td><td>Codec</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>LongSplat Ours</td><td>VVC</td><td>25.21 25.35</td><td>0.73 0.73</td><td>0.33 0.33</td><td>0.595 0.591</td><td>1.419 1.373</td><td>0.010 0.011</td></tr><tr><td>LongSplat Ours</td><td>H.264</td><td>24.80 24.95</td><td>0.71 0.72</td><td>0.34 0.34</td><td>0.971 0.856</td><td>1.818 1.589</td><td>0.016 0.014</td></tr><tr><td>LongSplat Ours</td><td>H.265</td><td>24.69 25.31</td><td>0.73 0.74</td><td>0.30 0.29</td><td>0.872 0.539</td><td>1.721 1.047</td><td>0.016 0.008</td></tr></table>

Table 8. Quantitative comparison on real-world compressed videos. All sequences are captured using a mobile device and evaluated under the same compression settings.
<table><tr><td></td><td>Scenes Method</td><td colspan="5">PSNRâ SSIMâ LPIPSâ RPEt â RPEr â ATEâ</td></tr><tr><td>Scenel</td><td>LongSplat Ours</td><td>35.60 35.68</td><td>0.96 0.96</td><td>0.08 0.08</td><td>0.062 0.061</td><td>0.086 0.077</td><td>0.001 0.001</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scene2</td><td>LongSplat Ours</td><td>31.43 31.47</td><td>0.86 0.86</td><td>0.09 0.09</td><td>0.135 0.124</td><td>0.190 0.165</td><td>0.002 0.002</td></tr></table>

<!-- image-->  
Figure 7. Qualitative results on real-world compressed videos. Comparison of rendered views between LongSplat and ours. Heatmaps show per-pixel deviations from GT, with warmer colors indicating larger differences.

## References

[1] Jose M Alvarez and Mathieu Salzmann. Compression-aware training of deep networks. Advances in neural information processing systems, 30, 2017. 3

[2] Anastasia Antsiferova, Sergey Lavrushkin, Maksim Smirnov, Aleksandr Gushchin, Dmitriy Vatolin, and Dmitriy Kulikov. Video compression dataset and benchmark of learning-based video-quality metrics. Advances in Neural Information Processing Systems, 35:13814â13825, 2022. 6

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022. 2

[4] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4160â4169, 2023. 2, 3, 6, 7, 8, 9

[5] Carlos Campos, Richard Elvira, Juan J Gomez Rodr Â´ Â´Ä±guez, Jose MM Montiel, and Juan D TardÂ´ os. Orb-slam3: An accu-Â´ rate open-source library for visual, visualâinertial, and multimap slam. IEEE transactions on robotics, 37(6):1874â 1890, 2021. 3

[6] Ali Mollaahmadi Dehaghi, Reza Razavi, and Mohammad Moshirpour. Reversing the damage: A qp-aware transformerdiffusion approach for 8k video restoration under codec compression. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1258â1267. IEEE, 2025. 3

[7] Longtao Feng, Xinfeng Zhang, Shanshe Wang, Yue Wang, and Siwei Ma. Coding prior based high efficiency restoration for compressed video. In 2019 IEEE International Conference on Image Processing (ICIP), pages 769â773. IEEE, 2019. 3

[8] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796â20805, 2024. 2, 3, 6, 7, 8, 9

[9] Jinpei Guo, Zheng Chen, Wenbo Li, Yong Guo, and Yulun Zhang. Compression-aware one-step diffusion model for jpeg artifact removal. arXiv preprint arXiv:2502.09873, 2025. 3

[10] Gang He, Guancheng Quan, Chang Wu, Shihao Wang, Dajiang Zhou, and Yunsong Li. Multi-frame deformable lookup table for compressed video quality enhancement. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3392â3400, 2025. 3

[11] Gang He, Weiran Wang, Guancheng Quan, Shihao Wang, Dajiang Zhou, and Yunsong Li. Rivuletmlp: An mlp-based architecture for efficient compressed video quality enhancement. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7342â7352, 2025. 6

[12] Xudong Huang, Wei Li, Jie Hu, Hanting Chen, and Yunhe Wang. Refsr-nerf: Towards high fidelity and super resolution view synthesis. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 8244â 8253, 2023. 3

[13] Bo Ji and Angela Yao. Sfm-free 3d gaussian splatting via hierarchical training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21654â21663, 2025. 2

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 4

[15] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36 (4):1â13, 2017. 2, 6, 7, 9

[16] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Ground- Ë ing image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71â91. Springer, 2024. 5, 2, 3

[17] Yinxiao Li, Pengchong Jin, Feng Yang, Ce Liu, Ming-Hsuan Yang, and Peyman Milanfar. Comisr: Compressioninformed video super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2543â2552, 2021. 3

[18] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5741â5751, 2021. 3

[19] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed 3d gaussian splatting for casual long videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27412â27422, 2025. 2, 3, 4, 5, 6, 7, 8, 9

[20] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, et al. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5166â5175, 2024. 2

[21] Xin Lin, Shi Luo, Xiaojun Shan, Xiaoyu Zhou, Chao Ren, Lu Qi, Ming-Hsuan Yang, and Nuno Vasconcelos. Hqgs: High-quality novel view synthesis with gaussian splatting in degraded scenes. In The Thirteenth International Conference on Learning Representations. 3

[22] Yike Liu, Jianhui Zhang, Haipeng Li, Shuaicheng Liu, and Bing Zeng. Coding-prior guided diffusion network for video deblurring. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10268â10277, 2025. 3

[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 2, 4

[24] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210â7219, 2021. 2

[25] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16539â16548, 2023. 6, 7, 8, 9

[26] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16539â16548, 2023. 2, 6, 7, 8

[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[28] Naama Pearl, Tali Treibitz, and Simon Korman. Nan: Noise-aware nerfs for burst-denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12672â12681, 2022. 3

[29] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016. 3

[30] Changha Shin, Woong Oh Cho, and Seon Joo Kim. Seam360gs: Seamless 360deg gaussian splatting from realworld omnidirectional images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28970â28979, 2025. 2

[31] Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the high efficiency video coding (hevc) standard. IEEE Transactions on circuits and systems for video technology, 22(12):1649â1668, 2012. 2

[32] Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2333â2343, 2025. 3

[33] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8248â8258, 2022. 2

[34] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558â16569, 2021. 3

[35] Gregory K Wallace. The jpeg still picture compression standard. Communications of the ACM, 34(4):30â44, 1991. 2

[36] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, and Shi-Min Hu. Nerf-sr: High quality neural radiance fields using supersampling. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6445â6454, 2022. 2, 3

[37] Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. F2-

nerf: Fast neural radiance field training with free camera trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4150â 4159, 2023. 2, 6, 7, 8, 3, 4

[38] Shiyao Wang, Hongchao Lu, and Zhidong Deng. Fast object detection in compressed video. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7104â7113, 2019. 3

[39] Yingwei Wang, Takashi Isobe, Xu Jia, Xin Tao, Huchuan Lu, and Yu-Wing Tai. Compression-aware video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2012â 2021, 2023. 3

[40] Thomas Wiegand, Gary J Sullivan, Gisle Bjontegaard, and Ajay Luthra. Overview of the h. 264/avc video coding standard. IEEE Transactions on circuits and systems for video technology, 13(7):560â576, 2003. 2

[41] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024. 2

[42] x265 Developers. x265 hevc encoder (version 7.0.2) [software]. https://www.x265.org/, 2025. Accessed: 2025-11-12. 6

[43] Mingyi Yang, Xile Zhou, Fuzheng Yang, Mingcai Zhou, and Hao Wang. Pimnet: A quality enhancement network for compressed videos with prior information modulation. Signal Processing: Image Communication, 117:117005, 2023. 3

[44] Ren Yang, Mai Xu, Zulin Wang, and Tianyi Li. Multi-frame quality enhancement for compressed video. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6664â6673, 2018. 3

[45] Jian Yue, Mao Ye, Luping Ji, Hongwei Guo, and Ce Zhu. A survey of deep-learning-based compressed video quality enhancement. IEEE Transactions on Broadcasting, 2025.

[46] Huimin Zeng, Jiacheng Li, and Zhiwei Xiong. Plug-and-play versatile compressed video enhancement. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17767â17777, 2025.

[47] Donghai Zhai, Xiaobo Zhang, Xun Li, Xichen Xing, Yuxin Zhou, and Changyou Ma. Object detection methods on compressed domain videos: An overview, comparative analysis, and new directions. Measurement, 207:112371, 2023. 3

[48] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341â359. Springer, 2024. 2

[49] Saiping Zhang, Luis Herranz, Marta Mrak, Marc GorrizÂ´ Blanch, Shuai Wan, and Fuzheng Yang. Dcngan: A deformable convolution-based gan with qp adaptation for perceptual quality enhancement of compressed video. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2035â2039. IEEE, 2022. 3

[50] Tingrong Zhang, Zhengxin Chen, Xiaohai He, Chao Ren, and Qizhi Teng. Qp-adaptive compressed video superresolution with coding priors. Signal Processing, 230: 109878, 2025. 3

[51] Xinjian Zhang, Su Yang, Wuyang Luo, Longwen Gao, and Weishan Zhang. Video compression artifact reduction by fusing motion compensation and global context in a swincnn based parallel architecture. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3489â 3497, 2023. 3

[52] Kun Zhou, Wenbo Li, Nianjuan Jiang, Xiaoguang Han, and Jiangbo Lu. From nerflix to nerflix++: A general nerfagnostic restorer paradigm. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3422â3437, 2023. 3

[53] Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, and Jiangbo Lu. Nerflix: High-quality neural view synthesis by learning a degradation-driven interviewpoint mixer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12363â12374, 2023. 3

[54] Qiang Zhu, Jinhua Hao, Yukang Ding, Yu Liu, Qiao Mo, Ming Sun, Chao Zhou, and Shuyuan Zhu. Cpga: Coding priors-guided aggregation network for compressed video quality enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2964â2974, 2024. 3

# CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video

Supplementary Material

This supplementary material provides additional implementation details, extended analyses of compressionrelated behavior, further experimental results across datasets and settings, and additional qualitative visualizations to complement the main paper.

## A. Implementation Details

Real-world Dataset Setup. We collected a real-world video dataset by recording two indoor scenes using an iPhone 17 Pro Max at 4K resolution and 30 fps. To ensure consistent training resolution, each captured frame was downsampled by a factor of 4Ã, resulting in input images at 1/4 of the original 4K resolution. Each captured sequence was then compressed using ffmpeg with the libx265 encoder, following the compression settings described above.

VVC Compression Setup. For the VVC experiments, we encoded each sequence using the official VVC reference software VTM 23.1 in the Random Access (RA) configuration. The RA setting was used with an intra period and GOP size of 32 frames, matching the temporal structure used in our HEVC/x265 experiments. We evaluated two compression levels using QP value of 37. The encoding was performed with the VTM EncoderApp and the standard encoder randomaccess vtm.cfg configuration file. After encoding, all bitstreams were decoded using DecoderApp to obtain the reconstructed YUV sequences, which were finally converted back into frame-wise RGB images using ffmpeg while preserving the original resolution and frame rate.

## B. Analysis

This section analyzes how compression-induced quality fluctuations manifest in real videos and provides empirical evidence for the issues identified in the main paper, thereby clarifying the motivation behind our compression-aware design. In this section, we focus on four aspects: (i) the impact of QP on input frames, (ii) frame-quality estimation based on QP and bitrate, (iii) the behavior of adaptive threshold adjustment for density control, and (iiii) the impact of PSNR gaps on feature matching stability.

Impact of QP on Input Frames. As shown in Fig. 8, the per-frame PSNR and the bitrate vary significantly with the chosen QP. This indicates that not only the overall visual quality of the input frames, but also the degree of frameto-frame quality variation within each sequence, depends strongly on the compression level. Fig. 9 illustrates the visual differences in the input frames at different QP values. As the QP increases from 27 to 47, the input frames become more heavily compressed, resulting in noticeable loss of fine details, increased blurring, and degradation in color fidelity.

<!-- image-->  
Figure 8. Frame-wise Y-PSNR and bits per frame of video sequences encoded at various QP values. Lower QP values yield higher PSNR and larger bit usage, while higher QP values result in reduced quality and substantially fewer coded bits. Periodic peaks correspond to I-frames within the GOP structure.

Analysis of QP and Bit Confidence Computation. Fig. 10 visualizes how QP and bit allocation fluctuate across the video sequence. The top and middle plots show that both QP values and bitrate vary substantially frame-by-frame, and their corresponding confidence scores $( q _ { t } ^ { q } , \ q _ { t } ^ { b } )$ follow these trends. This behavior aligns with typical codec operations, where high-QP or low-bit frames lose more highfrequency information, while low-QP or high-bit frames retain richer visual detail. The bottom plot shows the combined confidence $q _ { t } = q _ { t } ^ { q } + q _ { t } ^ { b }$ , which captures overall frame quality more reliably than either signal alone. High-quality frames consistently yield higher confidence scores, whereas strongly compressed frames produce lower scores, demonstrating that the unified confidence measure effectively reflects frame-wise quality fluctuations.

Analysis of Adaptive Threshold Adjustment. Fig. 11 visualizes how our adaptive density control responds to frame-level quality variations during training. In (a), the frame confidence $q _ { t }$ fluctuates according to the compression quality of each frame, while (b) shows its EMA-smoothed form $\bar { q } _ { t }$ , which captures the overall quality trend of the sequence. This smoothed baseline acts as a stable reference for modulating the strength of threshold adjustments. Plots (c) and (d) show the dynamic gradient threshold $\theta _ { t }$ and $\mathrm { d y } .$ namic opacity threshold $\omega _ { t } ^ { \prime }$ relative to their fixed originals (red lines). Both thresholds shift in accordance with the confidence signals, lowering or raising their values based on frame reliability.

<!-- image-->  
Uncompressed Image

<!-- image-->  
QP27

<!-- image-->  
QP37

<!-- image-->  
QP47

Figure 9. Effects of quantization parameter (QP) on video frames. As QP increases from 27 to 47, overall frame quality progressively degrades compared to the uncompressed reference.  
<!-- image-->  
Figure 10. Frame-wise QP, bitrate, and confidence scores. Top: per-frame QP values (blue) and the corresponding QP confidence $q _ { t } ^ { q }$ (red). Middle: per-frame bit allocation (orange) and the derived bitrate confidence $q _ { t } ^ { b }$ (blue). Bottom: combined confidence qt across the sequence. High-quality frames exhibit higher confidence values, while heavily compressed frames show consistently lower confidence.

<!-- image-->  
Figure 11. Analysis of adaptive threshold adjustment over training steps. Each plot shows how the corresponding quantity evolves across the first 100 training iterations: (a) frame confidence $q _ { t } ,$ , (b) EMA-smoothed confidence qÂ¯t, (c) dynamic gradient threshold Î¸t, and (d) dynamic opacity threshold $\omega _ { t } ^ { \prime } .$

Analysis of PSNR Gap Impact on Feature Matching. Fig. 12 illustrates the relationship between frame-wise PSNR gaps and the inlier ratio from MASt3R [16]. When large PSNR gaps occur between adjacent frames, the inlier ratio consistently drops, indicating degraded feature matching. This observation directly supports the second problem discussed in Sec. 3 of the main paper, where abrupt quality differences undermine pose initialization and destabilize view supervision. We also observe that such quality gaps negatively affect MASt3R correspondences themselves, as shown in Fig. 13, where compression reduces matching stability.

<!-- image-->  
Figure 12. Correlation between PSNR gap and inlier ratio. Larger frame-to-frame PSNR gaps (blue) coincide with drops in the MASt3R [16] inlier ratio (green), indicating that quality gaps directly deteriorate feature matching reliability.

<!-- image-->  
Figure 13. MASt3R [16] matching under compression. Feature correspondences become noticeably unstable when the input is compressed. The effect is most apparent in flat, low-texture regions (e.g., sky areas), where compression artifacts disrupt reliable matching and lead to incorrect alignments.

Table 9. Comparison of training efficiency on the Free dataset. All experiments were conducted on a single NVIDIA RTX 3080 GPU.
<table><tr><td>Method</td><td>FPSâ</td><td>Time (min)â</td><td>Size (MB)â</td></tr><tr><td>LongSplat [19]</td><td>40.89</td><td>117.07</td><td>82.46</td></tr><tr><td>Ours</td><td>40.86</td><td>116.62</td><td>82.44</td></tr></table>

## C. Additional Ablations

All additional ablation experiments in this section are evaluated on the Free dataset [37], with results averaged over seven scenes.

Table 10. Comparison of 2D refinement pipelines. We compare representative 2D and 3D refinement approaches in terms of rendering quality and runtime efficiency.
<table><tr><td>Pipeline</td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Runtimeâ</td></tr><tr><td>Deblurring 2D refinement</td><td>HQGS [21]</td><td>16.62</td><td>0.51</td><td>0.71</td><td>17m</td></tr><tr><td>2D refinement</td><td>PiSA-SR [32] + LongSplat</td><td>24.53</td><td>0.71</td><td>0.33</td><td>127m</td></tr><tr><td></td><td>CODiff [9] + LongSplat</td><td>23.52</td><td>0.67</td><td>0.37</td><td>149m</td></tr><tr><td>3D</td><td>Ours</td><td>25.31</td><td>0.74</td><td>0.29</td><td>117m</td></tr></table>

Table 11. Ablation study on confidence score formulations. We compare alternative scoring formulations and evaluate their impact on both photometric quality and pose accuracy.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>Non-linear (Eq. (4))</td><td>24.89</td><td>0.73</td><td>0.30</td><td>0.667</td><td>1.616</td><td>0.011</td></tr><tr><td>Non-exponential (Eq. (9))</td><td>24.94</td><td>0.74</td><td>0.29</td><td>0.672</td><td>1.483</td><td>0.013</td></tr><tr><td>Ours</td><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr></table>

Training Efficiency. Tab. 9 shows that our method incurs a slight reduction in FPS, while maintaining comparable training time and model size.

Comparison with 2D Refinement-Based Pipelines. Tab. 10 compares 2D and 3D refinement approaches on compressed video inputs. Existing 2D pipelines improve compressed frames using diffusion-based models such as PiSA-SR [32] and CODiff [9], and then apply LongSplat for reconstruction, resulting in a preprocessing strategy that is decoupled from 3D optimization. In contrast, our method directly performs compression-aware optimization in the 3D Gaussian domain and is specifically tailored for compressed videos. HQGS achieves faster runtime but assumes accurate camera poses and targets image-level degradations such as static JPEG compression, making it unsuitable for long unposed video sequences under real-world video compression, while diffusion-based 2D refinement methods incur high computational cost. These results demonstrate the benefit of direct 3D optimization over 2D refinement-based pipelines.

Ablation on Alternative Scoring Formulations. We investigate alternative scoring formulations to assess the impact of the functional design of frame confidence. Specifically, we evaluate a non-linear variant that replaces the linear mapping in Eq. (4) with a sigmoid-based formulation and a non-exponential variant that removes the exponential mapping in Eq. (9). For the non-linear variant, we apply a sigmoid-based scoring function to both the QP- and bitbased confidence terms. A temperature parameter Ï controls the smoothness of the score transition, while a scaling factor Ï restricts the contribution of the bit-based term; Ï denotes the sigmoid function. The resulting frame confidence is de-

Table 12. Ablation across different compression levels. We compare LongSplat and our method under QP settings of 27, 37, and 47 on the Free dataset, enabling evaluation of performance changes across varying degrees of compression.
<table><tr><td>Method</td><td>QP</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>LongSplat [19] Ours</td><td>27</td><td>25.54 25.56</td><td>0.75 0.75</td><td>0.30 0.30</td><td>0.631 0.742</td><td>1.426 1.405</td><td>0.011 0.012</td></tr><tr><td>LongSplat [19] Ours</td><td>37</td><td>24.69 25.31</td><td>0.73 0.74</td><td>0.30 0.29</td><td>0.872 0.539</td><td>1.721 1.047</td><td>0.016 0.008</td></tr><tr><td>LongSplat [19] Ours</td><td>47</td><td>23.72 23.80</td><td>0.64 0.64</td><td>0.44 0.44</td><td>0.842 0.963</td><td>1.829 1.958</td><td>0.013 0.017</td></tr></table>

fined as

$$
q _ { t } = \lambda ^ { q } \sigma ( ( \tilde { q } _ { t } - 0 . 5 ) / \tau ) + \lambda ^ { b } \rho \sigma \Big ( ( \tilde { b } _ { t } - 0 . 5 ) / \tau \Big ) .\tag{12}
$$

For the non-exponential variant, we change the exponential confidence modulation in Eq. (9) to a linear adjustment based on the confidence difference between the EMA baseline and the current frame. Specifically, the densification and pruning thresholds are modulated proportionally to the confidence gap $\left( \hat { q } _ { t } - q _ { t } \right)$ , rather than being scaled exponentially. This modification is formulated as:

$$
\theta _ { t } = \theta _ { 0 } \left( 1 + \alpha ( \bar { q } _ { t } - q _ { t } ) \right) , \qquad \omega _ { t } ^ { \prime } = \omega _ { 0 } \left( 1 + \alpha ( \bar { q } _ { t } - q _ { t } ) \right) ,\tag{13}
$$

where Î± is a scaling factor that controls the sensitivity of the threshold modulation to the confidence gap $\left( \hat { q } _ { t } - q _ { t } \right)$

As shown in Tab. 11, our default formulation with linear QPâbitrate aggregation and exponential mapping consistently outperforms alternative designs. Directly reflecting QP and bitrate in a linear manner provides a more faithful measure of frame reliability. In addition, the exponential mapping in our formulation smoothly amplifies confidence differences, whereas removing or altering this property can introduce abrupt or unstable confidence changes, which are detrimental to pose optimization.

Effect of Compression Levels. Tab. 12 presents the performance of LongSplat [19] and our method under different QP settings. Across all compression levels (QP 27, 37, and 47), our approach maintains stable reconstruction quality and pose accuracy, demonstrating robustness to varying degrees of compression applied to the Free dataset [37]. Fig. 14 further illustrates this trend, showing that our method preserves finer textures and clearer structures than LongSplat, especially as compression becomes stronger.

Evaluation with Compressed Video. Since uncompressed videos are unavailable in real-world scenarios, Tab. 13 reports results where all methods are evaluated on the encoded input sequences. Our method shows consistently higher performance across scenes compared to LongSplat.

QP 27  
QP 37  
QP 47  
GT  
<!-- image-->  
Figure 14. Rendering comparison between LongSplat and our method across different input video compression levels (QP 27, 37, 47). As QP increases, the input frames become more heavily compressed, leading to a noticeable decline in rendering quality for both methods. However, LongSplat exhibits stronger degradation, especially in text regions and fine details, which often appear blurred or broken. In contrast, our method adaptively grows or prunes Gaussians based on compression cues, enabling more faithful preservation of fine structures and clearer boundary reconstruction.

Table 13. Quantitative comparison on the Free dataset using compressed input frames. All methods are evaluated directly on codec-encoded sequences to reflect real-world conditions.
<table><tr><td rowspan="2">Scenes</td><td colspan="6">LongSplat [19]</td><td colspan="6">Ours</td></tr><tr><td>| PsNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â</td><td>RPEÏ âATEâ</td><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPEt â RPE, â ATEâ</td><td></td><td></td></tr><tr><td>Grass</td><td>24.51</td><td>0.67</td><td>0.34</td><td>0.092</td><td>1.270</td><td>0.006</td><td>24.69</td><td>0.68</td><td>0.33</td><td>0.322</td><td>2.115</td><td>0.007</td></tr><tr><td>Hydrant</td><td>24.31</td><td>0.77</td><td>0.25</td><td>0.815</td><td>0.838</td><td>0.022</td><td>23.96</td><td>0.75</td><td>0.26</td><td>0.963</td><td>1.306</td><td>0.031</td></tr><tr><td>Lab</td><td>26.09</td><td>0.85</td><td>0.18</td><td>1.230</td><td>1.499</td><td>0.016</td><td>26.06</td><td>0.85</td><td>0.18</td><td>1.249</td><td>1.497</td><td>0.016</td></tr><tr><td>Pillar</td><td>27.37</td><td>0.79</td><td>0.26</td><td>0.159</td><td>0.272</td><td>0.003</td><td>28.43</td><td>0.81</td><td>0.25</td><td>0.043</td><td>0.142</td><td>0.002</td></tr><tr><td>Road</td><td>21.59</td><td>0.71</td><td>0.35</td><td>2.660</td><td>5.882</td><td>0.035</td><td>21.67</td><td>0.71</td><td>0.35</td><td>2.122</td><td>3.524</td><td>0.027</td></tr><tr><td>Sky</td><td>25.84</td><td>0.85</td><td>0.20</td><td>0.467</td><td>1.521</td><td>0.011</td><td>26.30</td><td>0.86</td><td>0.20</td><td>0.364</td><td>1.521</td><td>0.007</td></tr><tr><td>Stair</td><td>27.46</td><td>0.83</td><td>0.24</td><td>0.135</td><td>0.205</td><td>0.003</td><td>27.77</td><td>0.83</td><td>0.23</td><td>0.134</td><td>0.165</td><td>0.002</td></tr><tr><td>Avg}$</td><td>25.31</td><td>0.78</td><td>0.26</td><td>0.794</td><td>1.641</td><td>0.014</td><td>25.55</td><td>0.78</td><td>0.26</td><td>0.742</td><td>1.467</td><td>0.013</td></tr></table>

Ablation on Parameter Î»q and $\lambda ^ { b } .$ . Tab. 14 analyzes the effect of the weighting coefficients Î»q and $\lambda ^ { b } .$ . Assigning a larger weight to the QP-based term consistently yields better photometric and pose accuracy, as QP directly reflects compression quality and frame reliability, whereas bitrate can vary due to scene complexity. Using a large bitrate weight degrades pose accuracy, and balanced weights lead to suboptimal performance. Accordingly, we adopt $\lambda ^ { q } ~ = ~ 1 . 0$ and $\lambda ^ { b } = 0 . 5$ , which achieves the best overall performance.

Table 14. Ablation on weighting coefficients $\lambda ^ { q }$ and $\lambda ^ { b }$ for frame-quality estimation. We study how the QP-based score weight (Î»q) and the bitrate-confidence score weight $( \lambda ^ { b } )$ affect both photometric quality and pose accuracy.
<table><tr><td> $\lambda ^ { q }$ </td><td> $\boldsymbol { \lambda } ^ { b }$ </td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td> $\mathrm { R P E } _ { t } \downarrow$ </td><td> $\mathrm { R P E } _ { r } \downarrow$ </td><td>ATEâ</td></tr><tr><td>0.5</td><td>0.5</td><td>24.96</td><td>0.74</td><td>0.30</td><td>0.875</td><td>1.353</td><td>0.015</td></tr><tr><td>0.5</td><td>1.0</td><td>24.81</td><td>0.73</td><td>0.30</td><td>0.743</td><td>1.589</td><td>0.013</td></tr><tr><td>1.0</td><td>0.5</td><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr><tr><td>1.0</td><td>1.0</td><td>24.94</td><td>0.74</td><td>0.29</td><td>0.822</td><td>1.705</td><td>0.013</td></tr></table>

Table 15. Ablation on momentum parameter $\beta$ evaluated on the Free dataset. Results are shown for different EMA momentum values, with the quality-gap parameter Î· fixed at 0.5.
<table><tr><td> $\beta$  PSNRâ</td><td>SSIMâ</td><td></td><td>LPIPSâ</td><td> $\mathrm { R P E } _ { t } \downarrow$ </td><td> $\mathrm { R P E } _ { r } \downarrow$ </td><td>ATEâ</td></tr><tr><td rowspan="2">0.9 0.95 (Ours)</td><td>24.80</td><td>0.73</td><td>0.30</td><td>0.762</td><td>2.133</td><td>0.013</td></tr><tr><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr><tr><td>0.99</td><td>24.95</td><td>0.73</td><td>0.30</td><td>0.667</td><td>1.401</td><td>0.012</td></tr></table>

Ablation on Momentum Parameter $\beta .$ Tab. 15 reports the effect of the momentum parameter $\beta .$ When $\beta$ is set to 0.9, the updates become relatively unstable, leading to lower reconstruction quality across PSNR, SSIM, and LPIPS. In contrast, assigning a very high value such as 0.99 results in excessive momentum accumulation, making the model less responsive to changes and degrading both reconstruction metrics and pose-related errors (RPE and ATE). When the momentum parameter is set to 0.95, the model achieves the best overall performance, providing a stable balance between responsiveness and smoothness. These findings suggest that choosing a well-balanced momentum value is essential for achieving stable updates and reliable performance throughout the pipeline.

Ablation on Quality-Gap Parameter $\eta \cdot$ Tab. 16 evaluates the effect of the pixel-drop scaling parameter $\eta ,$ which determines how strongly the inlier-ratio gap influences the masking applied to photometric supervision. Increasing Î· suppresses a larger portion of unreliable pixels, while smaller values reduce this effect. Among the tested settings, Î· = 0.5 provides the best overall performance.

## D. Dataset Construction Details: Video Compression Settings

We provide more detailed explanations of the video compression settings used in our experiments.

Table 16. Ablation on the quality-gap parameter Î·. Performance comparison on the Free dataset for different values of $\eta ,$ which controls the strength of quality-gap masking. All experiments fix the momentum parameter to $\beta = 0 . 9 5$
<table><tr><td>Î· PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td></td><td>RPEt â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>0.3</td><td>23.23</td><td>0.68</td><td>0.33</td><td>1.013</td><td>3.097</td><td>0.017</td></tr><tr><td>0.5 (Ours)</td><td>25.31</td><td>0.74</td><td>0.29</td><td>0.539</td><td>1.047</td><td>0.008</td></tr><tr><td>0.7</td><td>24.81</td><td>0.73</td><td>0.30</td><td>0.817</td><td>1.755</td><td>0.014</td></tr></table>

Encoder Configuration. For all image sequences used in the main paper, including the Tanks and Temples and the Free datasets, we encoded the input frames using ffmpeg and the libx265 at 60 fps. We employ the HEVC encoder libx265 (-c:v libx265) with the medium preset, -tune psnr, and the HEVC Main profile (-profile:v main). The maximum GOP length is set to 32 frames via -g 32. Additional encoder parameters are passed through -x265-params: qp enables constant-QP mode (we use several QP values, e.g., 27/37/47), keyint=32 and min-keyint=1 control the maximum and minimum I-frame interval, scenecut=40 activates moderate scene-cut detection, and open-gop=0 disables open GOPs to obtain a simpler temporal structure.

## Encoding Configuration

Various quality factors can be used by setting the QP value. For clarity and reproducibility, we provide the exact FFmpeg command used to produce the QP = 27 versions of the dataset. This command specifies consistent codec parameters such as GOP structure, chroma format, and logging full configuration is summarized in Tab. 17.

Logging and Quality Metrics. For each encoding, x265 writes per-frame statistics to a CSV file using csv=... and csv-log-level=1; these logs include the frame type, frame-level QP, number of bits, POC, encode order, and frame-level quality metrics (PSNR and SSIM for Y, U, V, and YUV channels). Adaptive quantization is disabled (aq-mode=0) so that the effective quantization closely follows the nominal QP. In addition to frame-level quantities, the CSV logs produced by x265 (csv-log-level=1) include several block-level coding statistics aggregated over all Coding Tree Units (CTUs) in each frame. These include the percentage of skip blocks and merge-mode blocks at multiple block sizes (64Ã64, 32Ã32, 16Ã16, 8Ã8), as well as the GOP position and temporal-layer index. These statistics provide a compact summary of the encoderâs intra/inter mode decisions and temporal prediction structure. Finally, all console output from ffmpeg (stdout and stderr) is redirected into a single log file for reproducibility.

Table 17. Encoding configuration. FFmpeg command used for x265 encoding.

ffmpeg -hide banner -y -f concat -safe 0 -r 60 -i input list.txt -vsync 0   
-vf "scale=trunc(iw/2)\*2:trunc(ih/2)\*2, format=yuv420p, setsar=1"   
-pix fmt yuv420p -c:v libx265 -preset medium -tune psnr -profile:v main -g 32   
-x265-params "qp=27:keyint=32:min-keyint=1:scenecut=40:open-gop=0:   
csv=logs/:sequence x265 qp27.csv:csv-log-level=1:psnr=1:ssim=1:aq-mode=0"   
outputs/sequence x265 qp27.mp4 > logs/sequence x265 qp27 encode.log 2>&1

## E. Complete Quantitative Evaluation

## E.1. Free Dataset

To complement the averaged pose metrics reported in the main paper, we provide scene-wise pose estimation results for the Free dataset [37] in Tab. 18. These results show that our method consistently improves both translational and rotational accuracy across all scenes.

## E.2. Tanks and Temples Dataset

Along with the averaged NVS reconstruction scores reported in the main paper, we include scene-wise NVS quality results for the Tanks and Temples dataset in Tab. 19. In addition, Tab. 20 presents the scene-wise camera pose estimation results, providing a more detailed analysis beyond the averaged metrics. Our method consistently improves both rendering fidelity and pose accuracy across individual scenes, demonstrating stronger robustness to compressionrelated degradation.

## E.3. Hike Dataset

To complement the averaged results reported in the main paper, we provide scene-wise novel view synthesis quality and camera pose estimation results on the Hike dataset in Tab. 21. Our method consistently improves both photometric quality and pose accuracy across most scenes compared to LongSplat, demonstrating robust performance under long trajectories, complex camera motion, and large-scale outdoor geometry.

## F. Additional Visual Comparisons

## F.1. Additional Free Dataset Results

Additional qualitative comparisons on the Free dataset [37] are presented in Fig. 15. Our method produces clearer object boundaries and alleviates blurring effectively, resulting in sharper and more structurally accurate reconstructions.

## F.2. Additional Tanks and Temples Dataset Results

Additional qualitative comparisons on the Tanks and Temples dataset [15] are shown in Fig. 16. Our method captures fine-grained details more faithfully and tends to mitigate blurring, resulting in more stable and detailed geometry.

## F.3. Additional Hike Dataset Results

The qualitative results in Fig. 17 further support these findings. While reconstructions obtained with LongSplat often exhibit blurry regions or unintended Gaussian artifacts in structurally complex areas such as dense vegetation, our method reconstructs much clearer and more stable geometry. Overall, our proposed approach demonstrates stronger robustness to compression-related degradation and preserves more consistent structural details even in such challenging datasets.

Table 18. Quantitative evaluation of camera pose estimation on the Free dataset [37]. We apply our method to both CF-3DGS [8] and LongSplat [19]. CF-3DGS fails to estimate a valid camera trajectory due to OOM when processing long sequences. Our method significantly reduces pose estimation errors for LongSplat across all scenes.
<table><tr><td rowspan="2">Scene</td><td colspan="2">[LongSplat (Uncomp.)</td><td colspan="2"></td><td colspan="2">NoPe-NeRF [4]</td><td colspan="2">CF-3DGS [8]</td><td colspan="2">CF-3DGS + Ours</td><td colspan="2">LongSplat [19]</td><td colspan="2">LongSplat + Ours</td></tr><tr><td>|RPEt âRPEr</td><td>â</td><td>ATEâ</td><td>|RPEt</td><td>âRPEr âATEâ</td><td>|RPE âRPEr</td><td>âATEâ</td><td>|RPE</td><td>â RPE âATE</td><td>|RPEt âRPEr</td><td></td><td>âATEâ</td><td>|RPEt âRPEr</td><td>âATEâ</td></tr><tr><td>Grass</td><td>0.037</td><td>0.255</td><td>0.001</td><td>3.631</td><td>10.256 0.353</td><td>OOM</td><td>OOM OOM</td><td>OOM</td><td>OOM OOM</td><td></td><td>0.092</td><td>1.270 0.006</td><td>0.061</td><td>0.550 0.002</td></tr><tr><td>Hydrant</td><td>1.161</td><td>1.419</td><td>0.029</td><td>5.198</td><td>4.990 0.523</td><td>1.163</td><td>7.550 0.042</td><td>1.161</td><td>7.908 0.043</td><td>1.101</td><td></td><td>1.155 0.034</td><td>0.044</td><td>0.367 0.002</td></tr><tr><td>Lab</td><td>1.161</td><td>2.163</td><td>0.013</td><td>6.186</td><td>2.597 0.489</td><td>OOM</td><td>OOM OOM</td><td>OOM</td><td>OOM OOM</td><td>1.491</td><td></td><td>1.744 0.023</td><td>1.249</td><td>1.497 0.016</td></tr><tr><td>Pillar</td><td>0.057</td><td>0.394</td><td>0.002</td><td>4.355</td><td>4.943 0.518</td><td>0.656</td><td>5.971</td><td>0.033 0.704</td><td>6.164 0.026</td><td>0.159</td><td>0.272</td><td>0.003</td><td>0.043</td><td>0.142 0.002</td></tr><tr><td>Road</td><td>2.018</td><td>5.738</td><td>0.026</td><td>6.511</td><td>3.655 0.433</td><td>OOM</td><td>OOM</td><td>OOM OOM</td><td>OOM</td><td>OOM 2.660</td><td>5.882</td><td>0.035</td><td>1.885</td><td>3.138 0.028</td></tr><tr><td>Sky</td><td>0.361</td><td>1.425</td><td>0.008</td><td>8.169</td><td>6.419 0.848</td><td>OOM</td><td>OOM OOM</td><td>OOM</td><td>OOM</td><td>OOM 0.467</td><td>1.521</td><td>0.011</td><td>0.364</td><td>1.521 0.007</td></tr><tr><td>Stair</td><td>0.137</td><td>0.187</td><td>0.002</td><td>6.791</td><td>0.1560.656</td><td>0.602</td><td>5.203 0.016</td><td>0.607</td><td>6.057</td><td>0.019 0.135</td><td>0.205</td><td>0.003</td><td>0.125</td><td>0.115 0.002</td></tr><tr><td>Avg.</td><td>0.705</td><td>1.654</td><td>0.012</td><td>5.834</td><td>4.717 0.546</td><td>0.807</td><td>6.241</td><td>0.030 0.824</td><td>6.710 0.029</td><td>0.872</td><td>1.721</td><td>0.016</td><td>0.539</td><td>1.047 0.008</td></tr></table>

Table 19. Quantitative evaluation of novel view synthesis quality on the Tanks and Temples dataset [15]. We apply our method to both CF-3DGS [8] and LongSplat [19]. Our method achieves improved rendering quality across most scenes. âUncomp.â denotes training and evaluation on original uncompressed videos, while other entries are trained on compressed videos and evaluated against uncompressed videos.
<table><tr><td rowspan="2">Scene</td><td colspan="3">[LongSplat (Uncomp.)</td><td colspan="3">NoPe-NeRF [4]</td><td colspan="3">CF-3DGS [8]</td><td colspan="3">CF-3DGS + Ours</td><td colspan="3">LongSplat [19]</td><td colspan="3">LongSplat + Ours</td></tr><tr><td>|PSNRâSSIMâLPIPSâ¥|</td><td></td><td></td><td>|PSNRâSSIMâLPIPSâ|</td><td></td><td></td><td>|PSNRâSSIMâLPIPSâ|PSNRâSSIMâLPIPSâ|</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>|PSNRâSSIMLPIPSâ|PSNRâSSIMâLPIPSâ</td><td></td></tr><tr><td>Church</td><td>32.31</td><td>0.94</td><td>0.10</td><td>24.01</td><td>0.70</td><td>0.44</td><td>27.32</td><td>0.83</td><td>0.26</td><td>28.39</td><td>0.83</td><td>0.31</td><td>28.84</td><td>0.85</td><td>0.25</td><td>28.92</td><td>0.85</td><td>0.25</td></tr><tr><td>Barn</td><td>27.53</td><td>0.79</td><td>0.21</td><td>28.48</td><td>0.82</td><td>0.33</td><td>25.53</td><td>0.72</td><td>0.33</td><td>27.60</td><td>0.85</td><td>0.21</td><td>27.51</td><td>0.78</td><td>0.29</td><td>28.12</td><td>0.79</td><td>0.29</td></tr><tr><td>Museum</td><td>25.00</td><td>0.78</td><td>0.14</td><td>28.84</td><td>0.85</td><td>0.27</td><td>27.05</td><td>0.80</td><td>0.25</td><td>26.99</td><td>0.80</td><td>0.25</td><td>29.12</td><td>0.85</td><td>0.22</td><td>29.14</td><td>0.85</td><td>0.22</td></tr><tr><td>Horse</td><td>27.96</td><td>0.89</td><td>0.12</td><td>25.52</td><td>0.82</td><td>0.32</td><td>28.65</td><td>0.86</td><td>0.22</td><td>27.43</td><td>0.83</td><td>0.26</td><td>29.45</td><td>0.87</td><td>0.21</td><td>29.48</td><td>0.87</td><td>0.21</td></tr><tr><td>Ballroom</td><td>26.85</td><td>0.88</td><td>0.15</td><td>26.25</td><td>0.78</td><td>0.33</td><td>27.63</td><td>0.84</td><td>0.21</td><td>29.70</td><td>0.83</td><td>0.29</td><td>25.91</td><td>0.82</td><td>0.24</td><td>26.17</td><td>0.82</td><td>0.24</td></tr><tr><td>Francis</td><td>31.66</td><td>0.90</td><td>0.17</td><td>28.17</td><td>0.83</td><td>0.40</td><td>29.48</td><td>0.82</td><td>0.29</td><td>28.98</td><td>0.86</td><td>0.22</td><td>29.98</td><td>0.82</td><td>0.29</td><td>30.04</td><td>0.83</td><td>0.29</td></tr><tr><td>Ignatius</td><td>30.26</td><td>0.91</td><td>0.11</td><td>26.41</td><td>0.77</td><td>0.36</td><td>25.87</td><td>0.71</td><td>0.35</td><td>25.80</td><td>0.71</td><td>0.35</td><td>26.64</td><td>0.73</td><td>0.35</td><td>26.72</td><td>0.73</td><td>0.35</td></tr><tr><td>Avg.</td><td>28.80</td><td>0.87</td><td>0.14</td><td>26.81</td><td>0.80</td><td>0.35</td><td>27.36</td><td>0.80</td><td>0.27</td><td>27.84</td><td>0.82</td><td>0.27</td><td>28.21</td><td>0.82</td><td>0.26</td><td>28.37</td><td>0.82</td><td>0.26</td></tr></table>

Table 20. Quantitative evaluation of camera pose estimation on the Tanks and Temples dataset [15]. We apply our method to both CF-3DGS [8] and LongSplat [19]. Our method achieves improved pose accuracy across most scenes. âUncomp.â denotes training and evaluation on original uncompressed videos, while other entries are trained on compressed videos and evaluated against uncompressed videos.
<table><tr><td rowspan="2">Scene</td><td colspan="3">|LongSplat (Uncomp.)</td><td colspan="2">NoPe-NeRF</td><td colspan="2">CF-3DGS</td><td colspan="3">CF-3DGS + Ours</td><td colspan="2">LongSplat</td><td colspan="2">LongSplat + Ours</td></tr><tr><td>|RPEt âRPEr âATEâ</td><td></td><td></td><td>|RPEt âRPEr âATEâ|</td><td></td><td>|RPEt âRPEr âATEâ|</td><td></td><td></td><td>|RPEt âRPEr âATEâ|</td><td></td><td></td><td>|RPEt âRPEr âATEâ|</td><td></td><td>|RPEt âRPEr âATEâ</td></tr><tr><td>Church</td><td>0.502</td><td>0.377</td><td>0.017</td><td>0.246</td><td>0.070 0.112</td><td>0.013</td><td>0.041 0.003</td><td></td><td>0.030</td><td>0.044 0.003</td><td>0.536</td><td>0.3760.018</td><td></td><td>0.506 0.379 0.016</td></tr><tr><td>Barn</td><td>0.654</td><td>1.302</td><td>0.009</td><td>0.027</td><td>0.017 0.003</td><td>0.048</td><td>0.142</td><td>0.007</td><td>0.029 0.022</td><td>0.003</td><td>0.847</td><td>0.341</td><td>0.013 0.594</td><td>0.280 0.009</td></tr><tr><td>Museum</td><td>1.809</td><td>1.995</td><td>0.010</td><td>0.202</td><td>0.243 0.019</td><td>0.054</td><td>0.147</td><td>0.006</td><td>0.054 0.148</td><td>0.006</td><td>0.809</td><td>0.500 0.007</td><td>0.867</td><td>0.523 0.007</td></tr><tr><td>Horse</td><td>1.141</td><td>1.891</td><td>0.012</td><td>1.245</td><td>0.135 0.030</td><td>0.129</td><td>0.093</td><td>0.005 0.013</td><td>0.041</td><td>0.003</td><td>0.992</td><td>0.715 0.010</td><td>0.875</td><td>0.686 0.009</td></tr><tr><td>Ballroom</td><td>7.549</td><td>2.212</td><td>0.035</td><td>0.060</td><td>0.027 0.003</td><td>0.029</td><td>0.021</td><td>0.003</td><td>0.041</td><td>0.174 0.008</td><td>7.852</td><td>2.745 0.051</td><td>7.763</td><td>2.747 0.049</td></tr><tr><td>Francis</td><td>1.314</td><td>0.979</td><td>0.032</td><td>0.534</td><td>0.213 0.035</td><td>0.041</td><td>0.174 0.008</td><td></td><td>0.129 0.093</td><td>0.005</td><td>4.435</td><td>0.881 0.108</td><td>4.227</td><td>0.877 0.103</td></tr><tr><td>Ignatius</td><td>0.530</td><td>0.354</td><td>0.012</td><td>0.047</td><td>0.008 0.005</td><td>0.047</td><td>0.074 0.008</td><td></td><td>0.047</td><td>0.074 0.008</td><td>0.387</td><td>0.259 0.009</td><td>0.401</td><td>0.267 0.009</td></tr><tr><td>Avg.</td><td>1.928</td><td>1.301</td><td>0.018</td><td>0.337</td><td>0.102 0.030</td><td>0.052</td><td>0.099</td><td>0.006</td><td>0.049 0.085</td><td>0.005</td><td>2.265</td><td>0.831 0.031</td><td>2.176</td><td>0.823 0.029</td></tr></table>

Table 21. Quantitative evaluation on the Hike dataset [25]. Our method consistently outperforms the LongSplat baseline across diverse scenes in terms of both photometric quality and pose accuracy. The results demonstrate improved robustness under long trajectories and challenging camera motions.
<table><tr><td rowspan="2">Scene</td><td colspan="6">LongSplat [19]</td><td colspan="6">LongSplat + Ours</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPE â</td><td> $\mathrm { R P E } _ { r } \downarrow$ </td><td>ATEâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>RPE â</td><td>RPEr â</td><td>ATEâ</td></tr><tr><td>forest1</td><td>18.78</td><td>0.45</td><td>0.37</td><td>0.447</td><td>5.938</td><td>0.009</td><td>19.14</td><td>0.48</td><td>0.36</td><td>0.457</td><td>5.513</td><td>0.009</td></tr><tr><td>forest2</td><td>24.96</td><td>0.76</td><td>0.20</td><td>0.421</td><td>0.908</td><td>0.012</td><td>25.98</td><td>0.81</td><td>0.17</td><td>0.332</td><td>0.558</td><td>0.011</td></tr><tr><td>forest3</td><td>15.00</td><td>0.38</td><td>0.45</td><td>1.560</td><td>12.286</td><td>0.067</td><td>15.11</td><td>0.39</td><td>0.45</td><td>1.551</td><td>11.158</td><td>0.067</td></tr><tr><td>garden1</td><td>23.32</td><td>0.73</td><td>0.20</td><td>0.375</td><td>0.999</td><td>0.010</td><td>23.37</td><td>0.73</td><td>0.20</td><td>0.492</td><td>0.944</td><td>0.011</td></tr><tr><td>garden2</td><td>17.57</td><td>0.36</td><td>0.46</td><td>2.287</td><td>11.308</td><td>0.034</td><td>17.67</td><td>0.37</td><td>0.45</td><td>2.320</td><td>10.880</td><td>0.033</td></tr><tr><td>garden3</td><td>19.42</td><td>0.43</td><td>0.41</td><td>1.169</td><td>6.450</td><td>0.015</td><td>19.96</td><td>0.47</td><td>0.38</td><td>0.846</td><td>5.453</td><td>0.015</td></tr><tr><td>indoor</td><td>17.74</td><td>0.63</td><td>0.42</td><td>0.959</td><td>9.635</td><td>0.033</td><td>17.27</td><td>0.62</td><td>0.42</td><td>0.761</td><td>7.311</td><td>0.031</td></tr><tr><td>playground</td><td>17.82</td><td>0.42</td><td>0.42</td><td>1.994</td><td>8.748</td><td>0.031</td><td>18.15</td><td>0.44</td><td>0.40</td><td>2.079</td><td>8.952</td><td>0.031</td></tr><tr><td>university1</td><td>19.22</td><td>0.48</td><td>0.41</td><td>1.485</td><td>4.852</td><td>0.029</td><td>19.12</td><td>0.47</td><td>0.43</td><td>1.554</td><td>5.068</td><td>0.023</td></tr><tr><td>university2</td><td>25.38</td><td>0.74</td><td>0.20</td><td>0.895</td><td>3.833</td><td>0.012</td><td>25.55</td><td>0.76</td><td>0.19</td><td>0.744</td><td>3.374</td><td>0.011</td></tr><tr><td>university3</td><td>18.23</td><td>0.49</td><td>0.38 0.45</td><td>1.545</td><td>7.978</td><td>0.017</td><td>19.05</td><td>0.53</td><td>0.35</td><td>1.462</td><td>8.415</td><td>0.015</td></tr><tr><td>university4</td><td>14.95</td><td>0.34</td><td></td><td>1.020</td><td>6.733</td><td>0.027</td><td>15.46</td><td>0.37</td><td>0.44</td><td>1.226</td><td>6.594</td><td>0.027</td></tr><tr><td>Avg.</td><td>19.37</td><td>0.52</td><td>0.36</td><td>1.180</td><td>6.639</td><td>0.025</td><td>19.65</td><td>0.54</td><td>0.35</td><td>1.152</td><td>6.185</td><td>0.024</td></tr></table>

<!-- image-->  
NoPe-NeRF

<!-- image-->  
LocalRF

<!-- image-->  
CF-3DGS

<!-- image-->  
LongSplat

<!-- image-->  
Ours

<!-- image-->  
GT  
Figure 15. Further qualitative comparisons on the Free dataset [37]. Compared to existing methods such as NoPe-NeRF [4], LocalRF [26], and CF-3DGS [8], our approach produces visually clearer and more stable reconstructions under compressed video inputs. Despite the blur and detail loss inherent in compressed frames, our method better preserves structural boundaries, reduces smoothing artifacts, and maintains sharper details across scenes. Furthermore, compared with LongSplat [19], our method explicitly considers the quality degradation caused by compression and thus enhances finer details, producing sharper and more expressive structures.

<!-- image-->  
NoPe-NeRF

<!-- image-->  
CF-3DGS

<!-- image-->  
LongSplat

<!-- image-->  
Ours

<!-- image-->  
GT

Figure 16. Further qualitative comparisons on the Tanks and Temples dataset [15]. Compared with existing methods such as NoPe-NeRF [4], CF-3DGS [8], and LongSplat [19], our method produces sharper and more stable reconstructions under compressed video inputs. Across diverse scenes, it better preserves structural boundaries and fine geometric details, mitigating the artifacts introduced by compression and resulting in more consistent and expressive 3D structures.  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 17. Qualitative results on the Hike dataset [25]. Compared with LongSplat [19], our method produces cleaner and more stable reconstructions across challenging outdoor scenes. In regions containing complex structures such as dense vegetation, LongSplat can exhibit blurring or produce unintended Gaussian artifacts in certain cases. In contrast, our method handles these situations more robustly overall and preserves more structurally consistent details.