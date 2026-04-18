# Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting

Semin Bae, Hansol Lim, Jongseong Brad Choi

Abstractâ The demand for large-scale digital twins is rapidly growing in robotics and autonomous driving. However, constructing these environments with 3D Gaussian Splatting (3DGS) usually requires expensive, purpose-built data collection. Meanwhile, deployed platforms routinely collect extensive omnidirectional RGB and LiDAR logs, but a significant portion of these sensor data is directly discarded or strictly underutilized due to transmission constraints and the lack of scalable reuse pipelines. In this paper, we present an omnidirectional RGB-LiDAR reuse pipeline that transforms these archived logs into robust initialization assets for 3DGS. Direct conversion of such raw logs introduces practical bottlenecks: inherent non-linear distortion leads to unreliable Structure-from-Motion (SfM) tracking, and dense, unorganized LiDAR clouds cause computational overhead during 3DGS optimization. To overcome these challenges, our pipeline strategically integrates an ERP-to-cubemap conversion module for deterministic spatial anchoring, alongside PRISMâa color stratified downsampling strategy. By bridging these multi-modal inputs via Fast Point Feature Histograms (FPFH) based global registration and Iterative Closest Point (ICP), our pipeline successfully repurposes a considerable fraction of discarded data into usable SfM geometry. Furthermore, our LiDAR-reinforced initialization consistently enhances the final 3DGS rendering fidelity in structurally complex scenes compared to vision-only baselines. Ultimately, this work provides a deterministic workflow for creating simulation-grade digital twins from standard archived sensor logs.

## I. INTRODUCTION

High-fidelity digital twins have become fundamental infrastructure for simulation, validation, and scenario generation in robotics and autonomous driving. While 3D Gaussian Splatting (3DGS) has proven highly effective for real-time view synthesis [1], constructing these environments traditionally relies on expensive, purpose-built data collection. Concurrently, deployed autonomous platforms routinely archive extensive volumes of omnidirectional RGB and LiDAR logs from everyday operations. Despite their high informational value, these data are predominantly underutilized due to a lack of scalable, automated reuse pipelines. Furthermore, direct RGB-only initialization from spherical imagery often suffers from geometric drift, and while LiDAR can reinforce geometric priors, adapting raw logs introduces practical bottlenecks such as excessive point-cloud density, crossmodal synchronization, and unstable alignment outcomes.

Consequently, the field currently lacks a standardized protocol to bridge the gap between archived sensor logs and robust neural scene reconstruction.

To address this, we present a reproducible and auditable omnidirectional RGB-LiDAR reuse pipeline. We focus on a highly practical question: how to convert existing, underused spherical logs into robust 3DGS initialization assets with minimal manual intervention. Our pipeline explicitly transforms raw logs through a systematic sequence of stages. First, we extract keyframes by evaluating omnidirectional image overlap via ORB features and homography, subsequently matching them with LiDAR scans using predefined temporal mappings. The selected images undergo ERP-tocubemap conversion to establish a robust spatial anchor through Structure-from-Motion (SfM) reconstruction. Concurrently, unaligned LiDAR sweeps are aggregated into a unified point cloud using Iterative Closest Point (ICP) based odometry [2]. This accumulated geometry is then colorized via sensor calibration data and substantially reduced using PRISM, a color-stratified downsampling strategy [3]. Finally, the downsampled LiDAR and SfM point clouds are aligned and fused using Fast Point Feature Histograms (FPFH) features [4] and ICP refinement [2], producing a comprehensive 3DGS-ready asset that integrates robust geometry, color data, and SfM points.

The primary goal of this work is to establish a deployable conversion workflow that effectively repurposes archived sensor logs. Beyond consistently improving the final 3DGS rendering fidelity, we comprehensively evaluate the systemâs end-to-end robustness. To this end, we report point-cloud reduction ratios, cross-modal alignment diagnostics (e.g., global fitness and ICP RMSE), and the resulting viewsynthesis performance across varying initialization densities. By generating deterministic, stage-level artifacts, we offer a practical and auditable pathway for creating simulation-grade digital twins from representative real-world sensor logs.

The main contributions of this paper are as follows:

1) We propose a deterministic, end-to-end data-reuse pipeline that effectively repurposes archived omnidirectional RGB-LiDAR logs into robust initialization assets for 3DGS, providing explicit reuse-efficiency accounting from raw sensor streams to usable SfM geometry.

2) We establish a robust modality-bridging workflow that strategically integrates temporal synchronization, ERPto-cubemap SfM spatial anchoring, ICP-based LiDAR aggregation, and PRISM-based color-stratified downsampling, overcoming inherent non-linear distortion and computational bottlenecks.

3) We present a comprehensive parameter sweep of the color-stratified downsampling strategy $\mathrm { ~  ~ { ~ \left( n \mathrm { ~  ~ { ~ \in ~ } } \right.}  ~ }$ $\{ 1 , 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \} \mathrm { \Omega }$ , providing detailed stage-level diagnostics to rigorously evaluate the robustness and limitations of cross-modal alignment.

4) We validate the proposed LiDAR-reinforced initialization against vision-only (vanilla) baselines, demonstrating consistent improvements in final 3DGS rendering fidelity in structurally complex scenes, while explicitly analyzing the quality-resource tradeoffs essential for practical digital-twin deployment.

## II. LITERATURE REVIEW

A. Neural Scene Representations and Initialization Bottlenecks

Neural Radiance Fields (NeRF) and its multiscale variants have established high-quality, anti-aliased novel-view synthesis for complex scenes [5]â[7]. To overcome the slow training and rendering speeds of implicit Multi-Layer Perceptron (MLP) architectures, subsequent works introduced explicit and hybrid data structures, such as voxel grids, hash encodings, and tensor cores [8]â[11]. Recently, 3D Gaussian Splatting (3DGS) has emerged as a highly efficient alternative, utilizing explicit anisotropic Gaussian primitives to achieve real-time rendering and rapid optimization [1]. This explicit representation has also inspired robotics-oriented extensions for downstream perception tasks [12].

However, the explicit nature of 3DGS introduces a critical vulnerability: its optimization is heavily dependent on the quality and density of the initial point cloud, typically derived from Structure-from-Motion (SfM) pipelines [13], [14]. Unlike implicit MLPs that can incrementally discover geometry from random initialization, 3DGS struggles to recover from unreliable or highly sparse SfM initializations, frequently resulting in severe floating artifacts or geometric collapse. Consequently, robust SfM point cloud generation has become a critical bottleneck for deploying 3DGS in large-scale or challenging autonomous driving environments.

## B. Omnidirectional Vision and Reconstruction Challenges

While recent neural rendering methods attempt to directly optimize radiance fields on spherical projections, they often struggle with severe non-linear distortion at the poles and uneven spatial sampling. More importantly for 3DGS, direct feature matching on raw Equirectangular Projection (ERP) imagery frequently fails in standard Structure-from-Motion (SfM) pipelines due to this inherent non-linear distortion, leading to highly sparse or inaccurate point cloud initialization [15]â[17]. To bypass this fundamental bottleneck, our pipeline incorporates an explicit ERP-to-cubemap conversion strategy. By transforming the spherical domain into six rectilinear, pinhole-equivalent faces, we enable robust feature matching, reliable camera pose tracking, and the dense point cloud initialization required for 3DGS.

## C. LiDAR Reinforcement in 3DGS

Vision-only 3D reconstruction fundamentally struggles with geometric ambiguities, most notably the absence of absolute metric scale and structural degradation in textureless or repetitive regions. To overcome these inherent limitations, recent works have increasingly incorporated LiDAR priors into 3DGS [18]â[21]. Methods such as LVI-GS, GS-SDF, and LIV-GS demonstrate that fusing precise, metric depth measurements from LiDAR significantly improves the structural integrity of Gaussian primitives. Furthermore, several approaches have successfully integrated LiDAR-camera setups for robust SLAM tracking and mapping [22]â[25].

Despite these advancements, existing integration methods predominantly focus on tightly-coupled, online SLAM pipelines, where 3DGS is optimized on-the-fly using perfectly synchronized, sequential sensor streams. In practical autonomous driving and robotics operations, however, extensive multi-sensor data is typically processed for offline trajectories and then merely archived. While previous largescale autonomous driving dataset literature focuses primarily on data creation, there is a stark absence of research addressing the automated conversion and reuse efficiency of these unstructured, archived logs into rendering assets. Furthermore, establishing a cross-modal alignment between these asynchronous modalities remains challenging; traditional registration techniques are prone to local minima when aligning physically distinct sensor modalities, such as scale-ambiguous, noisy SfM point clouds with dense LiDAR sweeps [2], [4]. Our work diverges from real-time view synthesis by proposing a deployable data-reuse pipeline that explicitly tackles cross-modal synchronization, robust globalto-local alignment (via FPFH and ICP), and the data-scale bottlenecks inherent in repurposing real-world logs.

## D. Point Cloud Sampling and Scale Reduction

While integrating LiDAR provides essential metric scale and structural priors, direct initialization of 3DGS with raw, accumulated LiDAR scans introduces practical computational bottlenecks. Fusing multiple sweeps generates tens of millions of points, and injecting such dense, unorganized point clouds into 3DGS leads to excessive memory footprints and over-parameterized Gaussian splitting during optimization. Traditional point cloud reduction techniques, such as uniform voxel grids or random sampling, often indiscriminately discard crucial high-frequency geometric features or minority color distributions vital for rendering fidelity.

Conversely, advanced sampling strategies like Farthest Point Sampling (FPS) or learning-based selection (e.g., PointNet++) offer superior geometric preservation but are computationally expensive when applied to large-scale archived logs [26]. To bridge this critical gap, our data-reuse pipeline integrates a PRISM-based color-stratified downsampling strategy [3]. By explicitly balancing geometric structure with calibrated RGB distributions, this approach effectively reduces point cloud density while preserving the essential multimodal priors. Ultimately, this scalable reduction enables the practical conversion of resource-intensive, discarded sensor logs into highly efficient 3DGS digital twin assets.

## III. METHODOLOGY

Figure 2 summarizes the proposed deterministic conversion workflow from archived omnidirectional logs to fused 3DGS initialization assets.

<!-- image-->  
Fig. 1. Example of ERP-to-cubemap conversion used in our pipeline. One omnidirectional panorama is projected into six rectilinear cubemap faces for robust feature matching in SfM.

## A. Modality Bridging: ERP-to-Cubemap Projection

Raw Equirectangular Projection (ERP) images inherently suffer from severe non-linear distortions at the poles, causing feature-matching failures in standard Structure-from-Motion (SfM). To ensure robust feature tracking, we explicitly project the ERP frames onto rectilinear cubemaps. A pixel $( u , v )$ in the ERP image corresponding to longitude $\theta \in$ $[ - \pi , \pi ]$ and latitude $\phi \in [ - \pi / 2 , \pi / 2 ]$ is mapped to a 3D ray d:

$$
\mathbf { d } = { \left[ \begin{array} { l } { \cos \phi \sin \theta } \\ { \sin \phi } \\ { \cos \phi \cos \theta } \end{array} \right] }\tag{1}
$$

Intersecting this ray with a unit cube restores rectilinear constraints, allowing standard multi-view geometry pipelines to reliably extract features. Figure 1 shows an example conversion from one ERP panorama to six rectilinear cubemap faces used in our SfM stage.

## B. Deterministic Spatial Anchoring via SfM

Using the undistorted cubemap faces, we establish a deterministic spatial anchor. For a 3D point $\mathbf { X } \in \mathbb { R } ^ { 3 }$ in the world coordinate, its projection u onto the i-th camera image plane is formulated as:

$$
s \tilde { \mathbf { u } } = \mathbf { K } _ { i } [ \mathbf { R } _ { i } \mid \mathbf { t } _ { i } ] \tilde { \mathbf { X } }\tag{2}
$$

where $\mathbf { K } _ { i }$ is the intrinsic matrix, and $\mathbf { R } _ { i } \in \mathrm { ~ \bf ~ \cal ~ S ~ } _ { \cal 0 } ( 3 )$ $\mathbf { t } _ { i } \in \mathbb { R } ^ { 3 }$ denote the camera extrinsics. To prevent the nondeterministic collapse common in noisy in-the-wild logs, we strictly constrain the optimization parameters in our SfM engine, yielding a scale-ambiguous sparse point cloud $\mathcal { P } _ { \mathrm { s f m } }$ and a set of reliable camera poses.

## C. LiDAR Colorization and PRISM Downsampling

Directly injecting massive LiDAR scans into the 3DGS optimizer causes severe VRAM bottlenecks. Conventional downsampling (e.g., Voxel Grid) enforces spatial uniformity, severely degrading semantic and photometric priors.

To resolve this, we colorize the LiDAR points using Eq. (2) and introduce PRISM (Color-Stratified Point Cloud Sampling). PRISM shifts the stratification domain from spatial coverage to visual complexity. Let C be the RGB color space partitioned into bins $B _ { c }$ . By imposing a maximum point capacity k per color bin, the downsampled set $\mathcal { P } _ { \mathrm { s u b } }$ is dynamically generated as:

$$
\mathcal { P } _ { \mathrm { s u b } } = \bigcup _ { c \in \mathcal { C } } \mathrm { S a m p l e } ( \mathcal { B } _ { c } , \ : \mathrm { m i n } ( \left| \mathcal { B } _ { c } \right| , k ) )\tag{3}
$$

This explicit preservation of chromatic diversity aggressively decimates visually homogeneous geometry while retaining the texture-rich regions crucial for spherical harmonics initialization.

## D. Robust Multi-Modal Alignment

The scale-ambiguous $\mathcal { P } _ { \mathrm { s f m } }$ must be metrically aligned with the downsampled LiDAR cloud $\mathcal { P } _ { \mathrm { s u b } }$ . Given the extreme noise and sparsity of in-the-wild data, global registration frequently diverges. Therefore, we bypass exhaustive global search and utilize a robust Iterative Closest Point (ICP) optimization initialized by trajectory metadata to find the optimal rigid transformation (Râ, tâ):

$$
\mathbf { R } ^ { * } , \mathbf { t } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { R } , \mathbf { t } } \sum _ { j = 1 } ^ { N } w _ { j } \left\| \mathbf { R } \mathbf { p } _ { \mathrm { s f m } } ^ { ( j ) } + \mathbf { t } - \mathbf { p } _ { \mathrm { s u b } } ^ { ( j ) } \right\| ^ { 2 }\tag{4}
$$

where $w _ { j }$ is a weight function to reject dynamic outliers. This local optimization effectively bridges the coordinate gap.

## E. 3DGS Initialization

The aligned, multimodal point cloud is directly transformed into 3DGS initialization assets. Each point $\mathbf { p } _ { i } \in \mathcal { P } _ { \mathrm { s u b } }$ initializes the mean $\pmb { \mu } _ { i }$ of a 3D Gaussian, formulated as:

$$
\mathcal { G } ( \mathbf { x } ) = \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } _ { i } ) ^ { T } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } _ { i } ) \right)\tag{5}
$$

The covariance matrix $\Sigma _ { i }$ is initialized based on local point density, and the projected RGB values initialize the zerothorder spherical harmonics. The registered extrinsics from Eq. (2) format the training views, establishing a deterministic foundation for 3DGS optimization.

<!-- image-->  
Fig. 2. Overview of the proposed omnidirectional RGB-LiDAR reuse pipeline. ERP images are converted to cubemaps for robust SfM anchoring, LiDAR odometry maps are colorized and PRISM-downsampled, and ICP-based registration fuses both modalities into a final 3DGS-ready asset.

## IV. EXPERIMENTAL SETUP AND IMPLEMENTATION

## A. System Implementation Details

Our data-reuse pipeline is designed to be highly memoryefficient, enabling the processing of massive multimodal sensor logs on a single workstation. All data preparation and 3D Gaussian Splatting (3DGS) rendering experiments were conducted on an NVIDIA RTX 4080 GPU with 16GB VRAM. This hardware constraint intentionally demonstrates that our PRISM-based reduction allows billion-point-scale logs to be processed without requiring enterprise-grade memory clusters. For software integration, we utilize the COLMAP framework for ERP-to-cubemap SfM sparse structure recovery and camera pose estimation [13], [14]. Crossmodal point cloud alignment (ICP) is implemented via the Open3D library. To isolate and evaluate the pure contribution of our multimodal initialization, all rendering tests utilize the original 3DGS CUDA implementation by Kerbl et al. [1] without any structural modifications to the training engine.

All preprocessing stages are executed with a shared configuration rather than per-scene retuning. In particular, the cubemap projection layout, SfM reconstruction policy, LiDAR color-transfer routine, and PRISM bucket sweep are kept fixed across all three sequences. This constraint is intentional: the target setting is archived-log conversion, so the pipeline is only practically useful if the same settings remain stable under different overlap patterns, texture conditions, and point-cloud densities.

## B. Dataset and Processing Protocol

To validate robustness and scalability across different campus trajectories, we evaluate our system on three largescale sequences from the AIR Lab omnidirectional RGB-LiDAR dataset.

AIR Lab 360 RGB-LiDAR dataset: We utilize three unstructured campus sequences: Dormitory 1, College of Engineering, and College of Physical Edu [27], [28]. This dataset features dense LiDAR paired with raw ERP (360- degree) imagery, providing an optimal stress test for our spherical-to-pinhole (ERP-to-cubemap) conversion module against severe non-linear lens distortion.

The selected sequences stress different failure modes for archived 3DGS initialization. Dormitory 1 contains repetitive facade structure and narrow road boundaries, College of Engineering introduces larger viewpoint change with mixed vegetation and vehicles, and College of Physical Edu contains broader open-space geometry with the largest accumulated clouds. Evaluating all three with the same sensor rig isolates algorithmic behavior from hardware variation and makes the later PRISM/ICP sweep directly comparable.

## C. Baseline Configurations and PRISM Sweep

To explicitly measure the impact of our LiDAR-reinforced initialization on 3DGS optimization, we establish the following experimental variants.

Vanilla (Baseline). A vision-only initialization trajectory utilizing only ERP keyframes, cubemap projection, and SfM sparse points, completely bypassing the LiDAR pipeline.

No-PRISM (Stress Case). A naive multimodal initialization that injects the fully densified, colorized LiDAR point cloud without downsampling. This serves as a qualitative failure case to demonstrate system collapse due to excessive memory footprint and over-parameterized Gaussian splitting.

Ours (n). The proposed end-to-end pipeline featuring PRISM color-stratified sampling. We conduct a parameter sweep across varying maximum points per RGB bucket, where n â {1, 5, 10, 20, 30, 50, 100}, to analyze the tradeoff between metric registration stability and rendering fidelity.

All 3DGS variants are trained with the same camera views, iteration budget, optimizer, and renderer. The only intentional difference is the initialization asset delivered to the trainer. This matched setup is important because it turns the comparison into an initialization study rather than a broader systemlevel benchmark confounded by view selection or training-

schedule changes.

## D. Evaluation Metrics and Artifact Contract

We evaluate both the efficiency of the data-reuse system and the final novel-view synthesis quality. Rendering fidelity is measured using PSNR, SSIM, and LPIPS after a fixed budget of 30,000 iterations. System-level robustness is evaluated through global alignment fitness, ICP RMSE, and the PRISM reduction ratio. Crucially, our pipeline enforces a strict artifact contract: every processing stage from LiDAR colorization to final SfM export automatically generates machine-readable JSON and CSV logs. This guarantees that the conversion of archived robot logs into 3DGS digital twins remains a fully deterministic, auditable, and reproducible engineering workflow.

## V. RESULTS AND DISCUSSION

## A. Pipeline Reuse and Throughput

Table I reports end-to-end conversion throughput and reuse proxies for the three AIR Lab sequences. Although

TABLE I  
PIPELINE SUMMARY WITH REUSE-EFFICIENCY PROXIES FOR THREE AIRLAB-360 SEQUENCES.
<table><tr><td>Sequence</td><td>ERP</td><td>LiDAR</td><td>Keyframes</td><td>SfM imgs</td><td>KF reuse</td><td>SfM rec.</td></tr><tr><td>Dormitory 1</td><td>280</td><td>280</td><td>103</td><td>509</td><td>36.8%</td><td>82.4%</td></tr><tr><td>College of Engineering</td><td>279</td><td>279</td><td>143</td><td>716</td><td>511.3%</td><td>83.4%</td></tr><tr><td>College of Physical Edu</td><td>479</td><td>479</td><td>170</td><td>907</td><td>35.5%</td><td>88.9%</td></tr></table>

the three sequences have different motion and coverage patterns, the same pipeline configuration runs without scenespecific branching. The keyframe reuse ratio (35.5%â51.3%) and SfM reconstruction ratio (82.4%â88.9%) show that a meaningful portion of archived logs can be converted into usable geometry without additional data collection. Table II complements this result with raw/SfM/PRISM point volumes and file sizes used for initialization.

These volume statistics also explain why later alignment behavior is not uniform across scenes. College of Physical Edu begins with the largest accumulated LiDAR and SfM clouds, indicating strong spatial coverage but also a harder reduction problem. By contrast, Dormitory 1 enters the sweep with fewer SfM points and more repetitive structure, making it more sensitive to how aggressively PRISM balances point density against stable cross-modal correspondence.

## B. PRISM Sweep and Alignment Behavior

Table III and Table IV summarize alignment quality and density reduction across the PRISM sweep (n â {1, 5, 10, 20, 30, 50, 100}).

As expected, larger n retains more points and lowers reduction ratios. Registration quality is generally strong, but not strictly monotonic: some scenes preserve high ICP fitness while RMSE degrades at dense settings, indicating that more points do not always translate to better local alignment. Averaging across scenes reinforces the same operating-region conclusion: moderate n balances compression and alignment more effectively, whereas very large n mainly adds optimization cost without guaranteed downstream benefit.

Across Tables IIâIV, the practical objective is not maximizing any single diagnostic in isolation but finding a density that remains robust across all stages. Low values of n offer strong compression and lighter downstream optimization, yet they can underrepresent the color diversity needed for stable Gaussian seeding. High values retain more structure, but once local correspondence becomes noisy the extra points mainly increase computational load instead of producing better rendering.

## C. 3DGS Quantitative Comparison

Table V and Table VI report absolute outcomes and deltas against the vanilla baseline after 30k iterations.

The LiDAR-initialized variants are clearly n-sensitive. Higher-density settings (n = 50, 100) consistently improve PSNR, while SSIM/LPIPS gains vary by scene. This pattern indicates that LiDAR priors are beneficial when cross-modal alignment quality is sufficiently reliable.

Resource cost follows the same direction as initialization density. Denser LiDAR priors increase training time, retained Gaussian count, and model size because the optimizer starts from a richer geometric prior and preserves more structure during refinement. The important practical result is that these overheads remain within a single-workstation budget, so the gains at n = 50 and n = 100 are achieved without assuming specialized multi-GPU infrastructure. Figure 3 shows the same trend qualitatively, with LiDAR-initialized runs recovering sharper boundaries and clearer local detail in thin branches and plate-like text.

## D. Discussion

Dormitory 1 and College of Engineering benefit more consistently from denser LiDAR priors, while College of Physical Edu remains harder due to broad open-space geometry and weaker fine-scale consistency. Across all three scenes, the main bottleneck is not pipeline executability but alignment reliability: global and ICP fitness can remain high even when rendering gains plateau, showing that correspondence quality matters more than raw point count alone.

For practical deployment, LiDAR reinforcement should be treated as conditional rather than universal. We recommend gating training on minimum registration quality, checking RGB-LiDAR projection overlays before 3DGS optimization, and retaining a vanilla fallback because the better initialization remains scene-dependent. All major claims are tied to machine-generated CSV/JSON artifacts under identical training budgets, so the reported metrics should be interpreted as deterministic initialization diagnostics rather than full heldout trajectory benchmarks.

## E. Limitations and Future Work

This study has four main limitations. First, residual spherical-image distortion still limits robust RGB color-to-LiDAR matching during LiDAR colorization in difficult regions. Second, experiments were conducted on only three outdoor sequences from the same sensor platform, so broader validation across more scenes, devices, and environmental conditions is still required. Third, we did not perform exhaustive parameter search for PRISM and ICP, so the current settings may be suboptimal for some trajectories. Fourth, the evaluation is offline and mostly static-scene oriented, leaving dynamic-object robustness and real-time deployment constraints for future work.

TABLE II  
POINT-CLOUD DATA SUMMARY USED FOR INITIALIZATION DIAGNOSTICS.
<table><tr><td>Sequence</td><td>Metric</td><td>Raw data</td><td>SfM</td><td>n = 1</td><td>n =5</td><td>n = 10</td><td>n = 20</td></tr><tr><td rowspan="2">Dormitory 1</td><td># points</td><td>2,058,126</td><td>82,724</td><td>145,437</td><td>426,720</td><td>628,446</td><td>878,976</td></tr><tr><td>Size (MB)</td><td>53.00</td><td>1.18</td><td>3.75</td><td>10.99</td><td>116.18</td><td>22.63</td></tr><tr><td rowspan="2">College of Engineering</td><td># points</td><td>2,275,569</td><td>88,903</td><td>146,667</td><td>442,041</td><td>660,160</td><td>942,330</td></tr><tr><td>Size MB)</td><td>58.59</td><td>1.27</td><td>3.78</td><td>11.38</td><td>17.00</td><td>24.26</td></tr><tr><td rowspan="2">College of Physical Edu</td><td># points</td><td>3,336,973</td><td>248,964</td><td>209,707</td><td>600,370</td><td>883,179</td><td>1,252,454</td></tr><tr><td>Size (MB)</td><td>85.92</td><td>3.56</td><td>5.40</td><td>15.46</td><td>22.74</td><td>32.25</td></tr></table>

TABLE III

ALIGNMENT DIAGNOSTICS GROUPED BY n (UPDATED SLAM-COLORIZED SWEEP).
<table><tr><td>n1</td><td colspan="3">Dormitory 1</td><td colspan="3">College of Engineering</td><td colspan="3">College of Physical Edu</td></tr><tr><td></td><td>Global â</td><td>ICP â</td><td>RMSEâ|</td><td>Global â</td><td>ICP â</td><td>RMSE â|</td><td>Global â</td><td>ICP â</td><td>RMSE â</td></tr><tr><td>1</td><td>0.9586</td><td>0.9874</td><td>0.3179</td><td>0.9944</td><td>0.9940</td><td>0.2640</td><td>0.9999</td><td>1.0000</td><td>0.1999</td></tr><tr><td>5</td><td>0.9505</td><td>0.9868</td><td>0.3171</td><td>0.9926</td><td>0.9948</td><td>0.2923</td><td>0.9922</td><td>0.9974</td><td>0.2200</td></tr><tr><td>10</td><td>0.8904</td><td>0.9867</td><td>0.3180</td><td>0.9127</td><td>0.9970</td><td>0.2076</td><td>0.9999</td><td>0.9999</td><td>0.2118</td></tr><tr><td>20</td><td>0.9275</td><td>0.9870</td><td>0.3185</td><td>0.9978</td><td>0.9985</td><td>0.2177</td><td>0.9997</td><td>1.0000</td><td>0.2138</td></tr><tr><td>30</td><td>0.9869</td><td>0.9876</td><td>0.2812</td><td>0.9941</td><td>0.9988</td><td>0.2004</td><td>0.9998</td><td>0.9983</td><td>0.2679</td></tr><tr><td>50</td><td>0.9122</td><td>0.9669</td><td>0.3817</td><td>0.9980</td><td>0.9995</td><td>0.2499</td><td>0.9897</td><td>0.9992</td><td>0.2436</td></tr><tr><td>100</td><td>0.8634</td><td>0.8946</td><td>0.3231</td><td>0.9923</td><td>0.9992</td><td>0.2006</td><td>0.9950</td><td>0.9999</td><td>0.2462</td></tr></table>

TABLE IV

PRISM OUTPUT SIZE AND REDUCTION RATIO GROUPED BY n.
<table><tr><td>n1</td><td colspan="2">Dormitory 1</td><td colspan="2">College of Engineering</td><td colspan="2">College of Physical Edu</td></tr><tr><td></td><td>Points</td><td>Reduction |</td><td>Points</td><td>Reduction</td><td>Points</td><td>Reduction</td></tr><tr><td>1</td><td>145,437</td><td>0.9293</td><td>146,667</td><td>0.9355</td><td>209,707</td><td>0.9372</td></tr><tr><td>5</td><td>426,720</td><td>0.7927</td><td>442,041</td><td>0.8057</td><td>600,370</td><td>0.8201</td></tr><tr><td>10</td><td>628,446</td><td>0.6947</td><td>660,160</td><td>0.7099</td><td>883,179</td><td>0.7353</td></tr><tr><td>20</td><td>878,976</td><td>0.5729</td><td>942,330</td><td>0.5859</td><td>1,252,454</td><td>0.6247</td></tr><tr><td>30</td><td>1,042,608</td><td>0.4934</td><td>1,130,716</td><td>0.5031</td><td>1,502,772</td><td>0.5497</td></tr><tr><td>50</td><td>1,258,801</td><td>0.3884</td><td>1,382,223</td><td>0.3926</td><td>1,841,923</td><td>0.4480</td></tr><tr><td>100</td><td>1,546,399</td><td>0.2486</td><td>1,723,438</td><td>0.2426</td><td>2,313,297</td><td>0.3068</td></tr></table>

TABLE V

ABSOLUTE 3DGS METRICS AT 30K ITERATIONS.
<table><tr><td>Sequence</td><td>Variant</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Train sec</td><td>Points</td><td>Model MB</td></tr><tr><td>Dormitory 1</td><td>Vanilla</td><td>27.4389</td><td>0.7817</td><td>0.3541</td><td>1030</td><td>1,205,789</td><td>1338</td></tr><tr><td>Dormitory 1</td><td>Ours (n = 5)</td><td>27.5551</td><td>0.7816</td><td>0.3536</td><td>1256</td><td>1,207,071</td><td>1326</td></tr><tr><td>Dormitory 1</td><td>Ours (n = 50)</td><td>27.7505</td><td>0.7854</td><td>0.3486</td><td>1611</td><td>1,542,370</td><td>1404</td></tr><tr><td>Dormitory 1</td><td>Ours (n = 100)</td><td>27.8043</td><td>0.7864</td><td>0.3473</td><td>1717</td><td>1,692,739</td><td>1439</td></tr><tr><td>College of Engineering</td><td>Vanilla</td><td>27.6077</td><td>0.7773</td><td>0.3538</td><td>626</td><td>1,054,880</td><td>1149</td></tr><tr><td>College of Engineering</td><td>Ours (n = 5)</td><td>27.7042</td><td>0.7747</td><td>0.3597</td><td>774</td><td>1,033,338</td><td>1141</td></tr><tr><td>College of Engineering</td><td>Ours (n = 50)</td><td>27.8964</td><td>0.7793</td><td>0.3526</td><td>1053</td><td>1,375,343</td><td>1218</td></tr><tr><td>College of Engineering</td><td>Ours (n = 100)</td><td>27.9105</td><td>0.7798</td><td>0.3515</td><td>1147</td><td>1,514,791</td><td>1252</td></tr><tr><td>College of Physical Edu</td><td>Vanilla</td><td>26.4412</td><td>0.7205</td><td>0.3672</td><td>700</td><td>1,219,200</td><td>1598</td></tr><tr><td>College of Physical Edu</td><td>Ours (n = 5)</td><td>26.3555</td><td>0.7018</td><td>0.3864</td><td>809</td><td>1,076,663</td><td>1515</td></tr><tr><td>College of Physical Edu</td><td>Ours (n = 50)</td><td>26.4927</td><td>0.7101</td><td>0.3783</td><td>1086</td><td>1,518,619</td><td>1634</td></tr><tr><td>College of Physical Edu</td><td>Ours (n = 100)</td><td>26.5048</td><td>0.7108</td><td>0.3769</td><td>1188</td><td>1,716,351</td><td>1679</td></tr></table>

<!-- image-->  
Fig. 3. Qualitative comparison across the three sequences. Columns show Vanilla, Ours (n = 50), and GT.

DELTA METRICS (LIDAR-INITIALIZED VARIANT - VANILLA) AT 30K ITERATIONS.
<table><tr><td>Sequence</td><td>Variant</td><td>âPSNR</td><td>ÎSSIM</td><td>âLPIPS</td></tr><tr><td>Dormitory 1</td><td>n = 5</td><td>+0.1162</td><td>-0.0001</td><td>-0.0005</td></tr><tr><td>Dormitory 1</td><td>n = 50</td><td>+0.3116</td><td>+0.0036</td><td>-0.0056</td></tr><tr><td>Dormitory 1</td><td>n = 100</td><td>+0.3654</td><td>+0.0047</td><td>-0.0068</td></tr><tr><td>College of Engineering</td><td>n = 5</td><td>+0.0966</td><td>-0.0026</td><td>+0.0059</td></tr><tr><td>College of Engineering</td><td>n = 50</td><td>+0.2887</td><td>+0.0020</td><td>-0.0012</td></tr><tr><td>College of Engineering</td><td>n = 100</td><td>+0.3028</td><td>+0.0025</td><td>-0.0022</td></tr><tr><td>College of Physical Edu</td><td>n = 5</td><td>-0.0857</td><td>-0.0187</td><td>+0.0192</td></tr><tr><td>College of Physical Edu</td><td>n = 50</td><td>+0.0516</td><td>-0.0105</td><td>+0.0112</td></tr><tr><td>College of Physical Edu</td><td>n = 100</td><td>+0.0636</td><td>-0.0097</td><td>+0.0097</td></tr></table>

## VI. CONCLUSION

We presented a deterministic omnidirectional RGB-LiDAR reuse pipeline that converts archived sensor logs into 3DGS-ready initialization assets. The reported reuse ratios and alignment/quality diagnostics show that substantial portions of previously underused logs can be repurposed into practical digital-twin inputs. Results across three real-world sequences further show that LiDAR reinforcement is effective but alignment-dependent, making robust cross-modal registration the key factor for consistent gains. Overall, this work provides an auditable baseline workflow for scalable digital-twin construction from existing field data.

## ACKNOWLEDGMENT

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2022-NR067080 and RS-2025- 05515607).

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[2] P. J. Besl and N. D. McKay, âA method for registration of 3-d shapes,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2, pp. 239â256, 1992.

[3] H. Lim, M. Im, and J. B. Choi, âPrism: Color-stratified point cloud sampling,â arXiv preprint arXiv:2601.06839, 2026.

[4] R. B. Rusu, N. Blodow, and M. Beetz, âFast point feature histograms (fpfh) for 3d registration,â in ICRA, 2009.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â European Conference on Computer Vision (ECCV), 2020.

[6] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf: A multiscale representation for anti-aliasing neural radiance fields,â in ICCV, 2021.

[7] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in CVPR, 2022.

[8] C. Sun, M. Sun, and H.-T. Chen, âDirect voxel grid optimization: Super-fast convergence for radiance fields reconstruction,â arXiv preprint arXiv:2111.11215, 2022.

[9] T. Mueller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â ACM Transactions on Graphics, vol. 41, no. 4, 2022.

[10] S. Fridovich-Keil, A. Yu, M. Chen, M. Tancik, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in CVPR, 2022.

[11] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensorf: Tensorial radiance fields,â in ECCV, 2022.

[12] J. W. Lee, H. Lim, S. Yang, and J. B. Choi, âMatt-gs: Masked attention-based 3dgs for robot perception and object detection,â arXiv preprint arXiv:2503.19330, 2025.

[13] J. L. Schoenberger and J.-M. Frahm, âStructure-from-motion revisited,â in CVPR, 2016.

[14] J. L. Schoenberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, âPixelwise view selection for unstructured multi-view stereo,â in ECCV, 2016.

[15] C. Geyer and K. Daniilidis, âA unifying theory for central panoramic systems and practical implications,â in ECCV, 2000.

[16] D. Scaramuzza, A. Martinelli, and R. Siegwart, âA toolbox for easily calibrating omnidirectional cameras,â in IROS, 2006.

[17] V. Usenko, N. Demmel, and D. Cremers, âThe double sphere camera model,â in 3DV, 2018.

[18] H. Lim, H. Chang, J. B. Choi, and C. M. Yeum, âLidar-3dgs: Lidar reinforcement for multimodal initialization of 3d gaussian splats,â Computers and Graphics, vol. 132, p. 104293, 2025.

[19] H. Zhao, W. Guan, and P. Lu, âLvi-gs: Tightly-coupled lidarvisual-inertial slam using 3d gaussian splatting,â arXiv preprint arXiv:2411.02703, 2024.

[20] J. Liu, Y. Wan, B. Wang, C. Zheng, J. Lin, and F. Zhang, âGssdf: Lidar-augmented gaussian splatting and neural sdf for geometrically consistent rendering and reconstruction,â arXiv preprint arXiv:2503.10170, 2025.

[21] R. Xiao, W. Liu, Y. Chen, and L. Hu, âLiv-gs: Lidar-vision integration for 3d gaussian splatting slam in outdoor environments,â arXiv preprint arXiv:2411.12185, 2024.

[22] J. Zhang and S. Singh, âLoam: Lidar odometry and mapping in realtime,â in Robotics: Science and Systems (RSS), 2014.

[23] T. Shan and B. Englot, âLego-loam: Lightweight and groundoptimized lidar odometry and mapping on variable terrain,â in IROS, 2018.

[24] T. Shan, B. Englot, C. Ratti, and D. Rus, âLio-sam: Tightly-coupled lidar inertial odometry via smoothing and mapping,â in IROS, 2020.

[25] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, âFast-lio2: Fast direct lidarinertial odometry,â IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053â2073, 2022.

[26] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, âPointnet++: Deep hierarchical feature learning on point sets in a metric space,â in NeurIPS, 2017.

[27] G. Kim, D. Son, S. Bae, K. Kim, Y. Jeon, S. Lee, S. Kim, S. Kim, J. Choi, J. Kwak, J. Choi, and J. Paik, âPair360: A paired dataset of high-resolution 360Â° panoramic images and lidar scans,â IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 9550â9557, 2024.

[28] Advanced Intelligence and Robotics Laboratory (AIR Lab), âAir lab 360 rgb-lidar dataset portal,â https://airlabkhu.github.io/ PAIR-360-Dataset/, 2024, accessed: 2026-03-05.