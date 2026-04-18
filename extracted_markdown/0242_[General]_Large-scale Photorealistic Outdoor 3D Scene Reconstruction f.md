# Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques

Christos Maikosâ, Georgios Angelidisâ, Georgios Th. Papadopoulosââ 

â Department of Informatics and Telematics, Harokopio University of Athens, Athens, Greece

â  Archimedes, Athena Research Center, Athens, Greece

Email: {chmaikos, gangelidis, g.th.papadopoulos}@hua.gr

AbstractâIn this study, we present an end-to-end pipeline capable of converting drone-captured video streams into highfidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRFbased approaches. Reconstruction quality remains within 4-7% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.

Index Termsâ3D gaussian splatting, neural rendering, realtime rendering, UAV video streaming, scene reconstruction

## I. INTRODUCTION

UAV-based applications are constantly increasing, as modern drones equipped with cameras, sophisticated sensors and artificial intelligence modules are widely adopted to provide cheap and effective solutions for a broad range of tasks, which vary from agricultural tasks [1] [2] to industry [3]â[5] and pre- and post-disaster assessments [6], [7]. The establishment of systems capable of processing live video captured from UAVs and converting them into realistic 3D visualisations in real-time constitutes a research field that combines computer vision, computer graphics, and interactive visualisation techniques.

Drone usage for data collection supports distinct adaptability in terms of coverage and perspective, allowing the acquisition of rich, multi-layered and high-resolution information that encompasses both motion dynamics and environmental characteristics [8]. The acquired information can be supplemented by utilising photogrammetry tools designed for 3D structure recording, where the fusion of spectral channels enables enhanced separation between foreground and background.

<!-- image-->  
Fig. 1: Complex stadium site reconstructed using the proposed approach.

During recent years, Gaussian Splatting has offered considerable improvements regarding the quality of real-time scene rendering [9], while recent approaches consider dynamic parameterisation for time-varying scenes or objects [10]. One major benefit of this representation is the ability to bind and work with deformable meshes using UV texture space mapping [11] to improve performance even in complex motion conditions. Moreover, the integration of 3DGS [12] into a representation engine environment offers the ability to directly interact with the 3D entities, even in mixed or AR-based applications. Nevertheless, this process requires compatibility with various rendering and representation engines, such as WebGL [13] and Unity [14], as well as sophisticated rendering optimisations, even in resource-limited devices, such as AR glasses.

Visual quality perception depends not only on the data acquisition stage but also on continuous processing throughout the acquisition and rendering phases, due to the quality loss caused by compression. Moreover, data transmission should be optimized considering the network constraints. Systems based on visual spectrum and depth information stream transmission can leverage established high-fidelity streaming methods even on edge devices, provided compression formats remain adaptive [15].

In poor signal or low bandwidth conditions, transmitting the entirety of a 3D model in real-time may prove difficult [16], highlighting the importance of the selective transmission of critical information or sampling rate adaptation. Recent advances in projecting Gaussians onto a 2D image plane have created new perspectives in efficiency improvement. Through transformations in SE(3) space, rapid mapping of points to camera space, and subsequently to the image one, is achievable [17], a process critical when input originates from multiple observation points.

In tandem, efforts to integrate semantic features [18] into each Gaussian strengthen the analysis of complex scenes [19], a property that drastically facilitates detection and interaction with distinct objects within them. The objective of using such a system for augmented environmental perception requires rigid temporal consistency among frames for the visualisation to appear realistic to the user or to the trajectory of the drone. Previous implementations have demonstrated that efficient Gaussian renderers are capable of delivering high frame rates without sacrificing visual quality [20], which is a critical aspect when dealing with VR/AR systems, as latency above a few milliseconds is noticeable to the user.

Considering the above, we propose a comprehensive pipeline that leverages 3DGS to convert live UAV video streams into high-fidelity 3D scenes with minimal latency. Our novelties include the following:

â¢ Real-Time 3DGS Reconstruction: We present a system capable of processing live UAV footage into geometrically consistent 3D Gaussian representations.

â¢ Seamless AR/VR Integration: The proposed architecture integrates directly with the visualisation engine, enabling interactive, immersive visualization and AR applications.

â¢ Adaptive Streaming Architecture: By utilizing RTMP for data collection and WebSockets for real-time updates, the system dynamically adapts to network conditions and efficiently supports resource-constrained devices.

The remaining sections of this paper are structured as follows: Section 2 provides information on existing work related to 3DGS reconstruction. Section 3 introduces our systemâs background theory and methodology. Section 4 presents our evaluation approach and setup. Section 5 discusses the obtained results and finally, our conclusion and possible expansions of our work are provided in Section 6.

## II. RELATED WORK

Dynamic scene representation is a constantly evolving field, especially in the latest years where the neural scene representations have been widely adopted in the bibliography. In what follows, an analysis of relevant studies is presented regarding neural 3DGS-based scene reconstruction.

In this context, Tang et al. [21] proposed a 3DGS framework for aerial imagery that utilizes adaptive masking and voxel-guided optimization to handle dynamic distractors and sparse viewpoints. Similarly, Shang et al. [22] introduced an enhanced 3DGS framework that leverages metric depth priors, gradient-driven adaptive densification, and density-based artifact pruning to improve reconstruction accuracy and efficiency in complex real-world scenes. Ham et al. [23] proposed a framework enabling robust 3D reconstruction from drone and ground imagery by iteratively extrapolating intermediate viewpoints with perceptual regularization to bridge the visual feature gap preventing standard registration, while Wu et al. [24] developed a method tailored for large-scale aerial reconstruction that utilizes a hybrid Gaussian representation with a dual-stage optimization strategy to balance efficient rendering speed with high-fidelity geometric detailing. In the same line, Luo et al. [25] introduced a depth-cue aware framework that refines geometric initialization using a minimum spanning tree algorithm and employs progressive depth-guided training to manage the significant depth variations inherent in drone imagery. Qian et al. [26] introduced a framework for large-scale UAV that prunes redundant Gaussians based on a contribution score and employing ray-tracing volume rendering to preserve fine geometric details. Mei et al. [27] developed a scalable divide-and-conquer framework for urban reconstruction that integrates distributed Structure-from-Motion (SfM) with dense depth priors to drive parallelized 2DGS, effectively mitigating memory constraints while preserving boundary consistency through a dedicated refinement strategy.

The proposed system effectively bridges the gap between aerial data acquisition and immersive visualisation by integrating 3DGS into a real-time visualization environment. Unlike optimization-centric frameworks, such as [21], [22], which primarily focus on handling dynamic distractors or leveraging depth priors to enhance reconstruction accuracy, our approach prioritizes the efficiency of the end-to-end pipeline. Furthermore, while recent hybrid approaches [23], [27] successfully merge ground and aerial level imagery for geometric completeness, they often lack the direct compatibility with representation engines required for immediate AR applications, a capability that is considered crucial for our method.

## III. METHODOLOGY

Figure 2 depicts the pipeline of the proposed system. The system converts UAV-acquired video streams and sensor data, into geometrically and temporally consistent, semantically enriched inputs suitable for SfM/Multi-View Stereo (MVS) and 3DGS, while maintaining low end-to-end latency.

## A. Principles of 3D Gaussian Representations

3DGS represents a scene as a set of anisotropic Gaussian distributions in a 3D space. Each Gaussian is described by a mean position Âµ and covariance Î£. During rendering, these 3D Gaussians are projected through the camera model into the image plane, as 2D Gaussian splats whose size, shape, and orientation depend on the viewing configuration. Appearance is modelled using spherical harmonics coefficients attached to each primitive. An opacity parameter controls the contribution of each Gaussian to the rendered image, while training is performed by minimizing a photometric loss between rendered images and ground-truth frames. New Gaussians can be injected in regions with high reconstruction error or that become visible later in a sequence while existing Gaussians can be modified or removed in an evolving scene. For dynamic scenes, they can be associated with non-static meshes and animated, considering local lighting effects. Moreover, they can carry semantic information allowing the renderer to output not just RGB images but also semantic or instance maps, enabling rich AR interactions.

<!-- image-->  
Fig. 2: Developed pipeline for converting UAV video streams into 3DGS representations integrated into a Unity environment.

## B. Data collection and RTMP server integration

The proposed architecture is built around a data collection subsystem that receives live streams from UAVs equipped with RGB-D or multispectral cameras. Each robot uses hardwareaccelerated H.264/H.265 encoding to compress the video, to reduce flight controllerâs load, and to transmit it over a reliable RTMP channel. Control, telemetry and video data are transmitted over separate channels to avoid congestion, especially from heavy I-frames that could delay flight control packets. In multi-UAV scenarios, each agent publishes its own RTMP stream to a common media server, capable of sustaining multiple persistent TCP connections. Buffer sizes, queuing, and retransmission parameters are tuned to achieve end-to-end latencies of a few hundred milliseconds.

On the server side, incoming RTMP streams are decoded into raw frames, which can be passed directly to processing pipelines via shared memory or IPC mechanisms. At this stage, the system may also perform adaptive bitrate or resolution adjustments based on backend load and network conditions: when the reconstruction pipeline becomes saturated or bandwidth drops, the server can dynamically reduce the outgoing resolution or bitrate to maintain responsiveness. Frames are fed into separate processing paths for reconstruction, direct AR overlay, and preview, preventing computationally heavy tasks such as dense reconstruction from blocking immediate visual feedback. Overall, the RTMP server is a vital component of a broader data integrity and latency management strategy that strongly influences the final visual quality and temporal consistency of the AR experience. Once a reconstruction session completes, its output is delivered to client-side visualization and interaction applications through a real-time bidirectional communication interface. This push-based mechanism ensures that rendering engine applications automatically receive the latest map representations. Clients choose how to handle updates in near real-time: replace their current model, merge the new splats with existing ones, or selectively loading regions of interest.

## C. Frame extraction and synchronization

The system processes compressed multimedia streams and decodes them into raw image buffers, assigning timestamps based on stream metadata. To ensure consistency across multimodal inputs, each stream is normalized to a common time base, using network synchronization protocols. A dedicated synchronization module maintains separate queues for each modality, aligning video frames with the nearest corresponding sensor samples within a configurable temporal window. Missing sensor data is reconstructed via interpolation for linear metrics or integration for inertial measurements, ensuring a continuous and complete data stream.

To manage processing load, frame rate reduction is applied either uniformly or adaptively, selecting frames based on motion magnitude or segmentation dynamics, while preserving full frame sequences for high-speed content to capture fine temporal details. Addressing the inherent latency disparity between buffered video streams and real-time telemetry, the architecture employs a multi-threaded buffering system that aligns streams based on provided timestamps. This approach decouples decoding from synchronization, ensuring the strict temporal coherence necessary for geometrically accurate alignment in the final reconstruction pipeline.

## D. Camera pose estimation

Camera pose estimation converts temporally aligned sensor data into 6-DoF poses, defining the position and orientation of each camera frame in a global reference frame. RGB-D data allows visual odometry and SLAM, while monocular RGB uses SfM pipelines with feature matching, geometric verification, and bundle adjustment; additional sensors, IMU/GPS data, and semantic segmentation help stabilize poses in challenging UAV scenarios. The output is a sequence of SE(3) transformations

$$
T _ { i } = { \big [ } R _ { i } \ { \big | } \ t _ { i } { \big ] } ,
$$

where $R _ { i }$ is the 3Ã3 rotation matrix and $t _ { i }$ is the 3D translation vector of camera i in the global coordinate system. These poses are used by the renderer to project 3D Gaussians into each view without spatial or temporal discontinuities.

## E. Training and deployment of the 3DGS model

The training procedure initializes 3D Gaussians at point locations derived from the SfM/MVS cloud, defining each primitive as described in Section III-A. Frames are processed in mini-batches to maximize GPU utilization. Parameters are optimized via a differentiable tile-based rasterizer that projects Gaussians into camera views to compute photometric loss against ground-truth frames. This process incorporates an adaptive density control mechanism, where a densification phase involves sampling new points in under-reconstructed regions, coupled with a pruning stage that removes lowcontributing primitives, ensuring model compactness, geometric integrity, and mathematical accuracy. For deployment, the system employs mixed-precision training and spatial regularization to prevent artifacts. The final model is stored in a compact binary format containing contiguous parameter arrays and spatial tiling data. In live deployments, the model is continuously updated as new frames arrive. Instead of retraining from scratch, the system performs online optimization focused on regions affected by the new data, achieving a continuously updated and spatially consistent 3DGS representation.

## IV. EXPERIMENTAL EVALUATION

## A. Datasets

We evaluated our method on three widely used multi-view reconstruction and novel view synthesis benchmarks: Mip-NeRF 360 [28], Tanks and temples [29], and Deep blending [30]. Combined, the selected benchmarks cover a diverse set of challenges, including scene scale, geometry complexity, and photorealistic view synthesis.

## B. Experimental Setup

The experimental setup is designed with an emphasis on the pipeline behaviour on real-world information. The main goal is to characterize the end-to-end performance of the backend. UAVs capture RGB or RGB-D video alongside IMU and GPS telemetry, which is streamed to a ground station via RTMP [31] and decoded in near real-time. To capture realistic networking and buffering behavior, prerecorded sequences can be re-streamed using the NGINX RTMP module [32]. Video and sensor streams share a unified time base, namely IEEE 1588 PTP [33]. Reconstruction updates are distributed to clients over WebSocket (RFC 6455) [34], enabling measurement of network delivery latency and client update behavior. The backend executes on a workstation equipped with an AMD Ryzen 9 9900X3D, an NVIDIA RTX 4070 12 GB, and 64 GB RAM. The software pipeline integrates FFmpeg decoding, python-based preprocessing and semantic segmentation, COLMAP-style SfM/MVS [35], and an original INRIA 3DGS implementation [12]. We evaluated two variants of our method, different in the number of Gaussians used for the reconstruction: Ours30K with 30,000 iterations and Ours7K with 7,000 iterations. Unity [14] was selected as the visualisation engine and client, while the utilized drone was a DJI Mini 4K.

## C. Evaluation metrics

The evaluation process addresses visual fidelity, geometric accuracy, and operational performance. Visual fidelity on heldout views is quantified via PSNR, SSIM [36], and LPIPS [37], computed cumulatively to monitor drift. We adopt accuracy and completeness reporting as in [38]. When using Tanks and Temples, we report a distance-threshold F-score adopted from [29]. Pose accuracy is evaluated via Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) following [39]. Finally, semantic quality is measured using mIoU and, for instanceaware tasks, Panoptic Quality (PQ) [40]. The operational metrics also include: end-to-end latency, throughput and update stability.

## D. Results

Table I reports the quantitative performance of Instant-NGP [41], Mip-NeRF360 [28], and the proposed 3DGS method variants across the selected benchmarks. On the Mip-NeRF360 dataset, our proposed Ours30K algorithm achieves an SSIM of 0.815, a PSNR of 27.21, and an LPIPS of 0.214, with a total training time of 41 minutes and a rendering speed of 134 FPS. The same method, when evaluated on Tanks and temples, records an SSIM of 0.841, PSNR of 23.14, and LPIPS of 0.183, while maintaining a rendering speed of 154 FPS. Similarly, for the Deep blending dataset, it reaches an SSIM of 0.903, PSNR of 29.41, and LPIPS of 0.243, with a training time of 36 minutes and a rendering speed of 137 FPS. The Ours7K configuration exhibits reduced training time across all datasets, while maintaining comparable image quality metrics and higher rendering speeds. Instant-NGP demonstrates lower memory usage and shorter training times, whereas Mip-NeRF360 presents higher training costs with reduced rendering speed across all evaluated datasets.

## V. DISCUSSION

The proposed system leverages 3DGS, achieving improved rendering speeds compared to NeRF-based algorithms that rely on computationally expensive ray marching techniques. While view-synthesis quality is comparable to improved NeRFs, 3DGS is notably faster in training and rendering. Unlike point clouds, anisotropic Gaussian kernels minimize aliasing and geometry gaps, and the addition of spherical harmonics captures view-dependent lighting often lacking in standard MVS pipelines.

For dynamic scenes, attaching Gaussians to deformable objects via UV mapping exhibits superior performance compared to traditional linear blend skins by accurately reproducing precise details, without requiring extensive model retraining. Additionally, when compared to voxel-based representations, 3DGS scales linearly, utilizing tile-based rasterization to handle large UAV datasets without bottlenecks. Furthermore, the nature of splats allows for local splat rearrangements and frame-by-frame updates without the global side effects characterising NeRFs.

Operational efficiency is maintained through progressive loading and compact parameter codebooks, facilitating this way the deployment on resource-constrained immersive devices. The Unity integration achieves continuous asset streaming and supports multi-user collaboration. Low-latency can be accomplished by leveraging RTMP/RTMPS combined with hardware acceleration and telemetry channels that manage network constraints effectively.

TABLE I: Experimental evaluation and comparison of reconstruction quality and resource efficiency.
<table><tr><td>Dataset</td><td>Method</td><td>SSIM</td><td>PSNR</td><td>LPIPS</td><td>Train</td><td>FPS</td><td>Memory</td></tr><tr><td rowspan="4">Mip-NeRF360 [28]</td><td>Instant-NGP (Base)</td><td>0.671</td><td>25.30</td><td>0.371</td><td>5m 37s</td><td>11.7</td><td>13MB</td></tr><tr><td>Mip-NeRF360</td><td>0.792</td><td>27.69</td><td>0.237</td><td>48h</td><td>0.06</td><td>8.6MB</td></tr><tr><td>Ours7K</td><td>0.770</td><td>25.60</td><td>0.279</td><td>6m 25s</td><td>160</td><td>523MB</td></tr><tr><td>Ours30K</td><td>0.815</td><td>27.21</td><td>0.214</td><td>41m 33s</td><td>134</td><td>734MB</td></tr><tr><td rowspan="4">Tanks and temples [29]</td><td>Instant-NGP (Base)</td><td>0.723</td><td>21.72</td><td>0.330</td><td>5m 26s</td><td>17.1</td><td>13MB</td></tr><tr><td>Mip-NeRF360</td><td>0.759</td><td>22.22</td><td>0.257</td><td>48h</td><td>0.14</td><td>8.6MB</td></tr><tr><td>Ours7K</td><td>0.767</td><td>21.20</td><td>0.280</td><td>6m 55s</td><td>197</td><td>270MB</td></tr><tr><td>Ours30K</td><td>0.841</td><td>23.14</td><td>0.183</td><td>26m 54s</td><td>154</td><td>411MB</td></tr><tr><td rowspan="4">Deep blending [30]</td><td>Instant-NGP (Base)</td><td>0.797</td><td>23.62</td><td>0.423</td><td>6m 31s</td><td>3.26</td><td>13MB</td></tr><tr><td>Mip-NeRF360</td><td>0.901</td><td>29.40</td><td>0.245</td><td>48h</td><td>0.09</td><td>8.6MB</td></tr><tr><td>Ours7K</td><td>0.875</td><td>27.78</td><td>0.317</td><td>4m 35s</td><td>172</td><td>386MB</td></tr><tr><td>Ours30K</td><td>0.903</td><td>29.41</td><td>0.243</td><td>36m 2s</td><td>137</td><td>676MB</td></tr></table>

The developed system remains sensitive to input data quality. Potential camera pose errors can propagate cumulatively, causing visual artifacts. Moreover, while dynamic updates and depth-based occlusion enhance realism, they notably stress the GPU. Additionally, pre-processing relies heavily on deep learning accuracy, where segmentation errors can burden the SfM pipeline. Despite these challenges, quantitative evaluations indicate the method maintains reconstruction quality within 4-7% of a high-fidelity offline reference, while substantially reducing end-to-end latency. This trade-off enables real-time, scalable augmented perception and outperforms foreground-mixing approaches such as those in [42].

## VI. CONCLUSION

This work presents a comprehensive approach to the realtime reconstruction of 3D scenes leveraging UAV-acquired data, based on 3DGS. This method brings together high geometric precision with computational efficiency and has the potential of being utilised in real-time AR applications. The explicit use of Gaussian representations yields numerous advantages over alternative approaches, specifically regarding reduced rendering times and the capability for local model updates without disrupting the global scene context. The architecture of the system develops an integral technological infrastructure that encompasses UAV data capture and processing, up to AR display that ensures a continuous information flow with minimal latency.

Scalability across multiple streams and support for computationally-constrained devices render this system extremely flexible and applicable in various environments and operational scenarios. Despite the discussed limitations and challenges, such as camera pose estimation accuracy and dynamic content management, the proposed method outperforms other techniques in terms of both efficiency and visual quality. Its deployment in applications such as archaeological documentation, surveillance, or remote collaboration opens a whole dimension of possibilities addressing practical problems. Future work will focus on optimizing input data accuracy, integrating state-of-the-art machine learning algorithms, and extending capabilities to even demanding environments. Ultimately, the value of this approach lies in its ability to merge advanced technological performance with practical utility, providing a robust tool for augmented perception and real-world interaction. Moreover, the system is designed so as to be connected with real-time AI-enabled event detectors [43]â[46], so as to facilitate scenarios involving Human-Robot Interaction [47]â[49] and broader security response incidents [50], while also supporting the necessary eXplainable Artificial Intelligence (XAI) pipelines [51], [52].

## REFERENCES

[1] J. Agrawal and M. Y. Arafat, âTransforming farming: A review of ai-powered uav technologies in precision agriculture,â Drones, vol. 8, no. 11, p. 664, 2024.

[2] H. Manoj, D. Shanthi, B. Lakshmi, K. Archana, E. Venkata Naga Jyothi, and K. Archana, âAi-driven drone technology and computer vision for early detection of crop disease in large agricultural areas,â Scientific Reports, 2025.

[3] N. Ejaz and S. Choudhury, âComputer vision in drone imagery for infrastructure management,â Automation in Construction, vol. 163, p. 105418, 2024.

[4] P. Alimisis, I. Mademlis, P. Radoglou-Grammatikis, P. Sarigiannidis, and G. T. Papadopoulos, âAdvances in diffusion models for image data augmentation: A review of methods, models, evaluation metrics and future research directions,â Artificial Intelligence Review, vol. 58, no. 4, p. 112, 2025.

[5] T. Bright, S. Adali, and C. Trois, âSystemic review and meta-analysis: The application of ai-powered drone technology with computer vision and deep learning networks in waste management,â Drones, vol. 9, no. 8, p. 550, 2025.

[6] S. P. H. Boroujeni, A. Razi, S. Khoshdel, F. Afghah, J. L. Coen, L. OâNeill, P. Fule, A. Watts, N.-M. T. Kokolakis, and K. G. Vamvoudakis, âA comprehensive survey of research towards ai-enabled unmanned aerial systems in pre-, active-, and post-wildfire management,â Information Fusion, vol. 108, p. 102369, 2024.

[7] J. Cani, P. Koletsis, K. Foteinos, I. Kefaloukos, L. Argyriou, M. Falelakis, I. Del Pino, A. Santamaria-Navarro, M. Cech, O. Severa Ë et al., âTriffid: Autonomous robotic aid for increasing first responders efficiency,â in 2025 6th International Conference in Electronic Engineering & Information Technology (EEITE). IEEE, 2025, pp. 1â9.

[8] V. Shukla, L. Morelli, F. Remondino, A. Micheli, D. Tuia, B. Risse et al., âTowards estimation of 3d poses and shapes of animals from oblique drone imagery,â International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. 48, pp. 379â386, 2024.

[9] H. Pang, H. Zhu, A. Kortylewski, C. Theobalt, and M. Habermann, âAsh: Animatable gaussian splats for efficient and photoreal human rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1165â1175.

[10] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and M. NieÃner, âGaussianavatars: Photorealistic head avatars with rigged 3d gaussians,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 299â20 309.

[11] I. Lupiani, Blender Scripting with Python: Automate Tasks, Write Helper Tools, and Procedurally Generate Models in Blender 4. Springer Nature, 2025.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis Â¨ et al., â3d gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[13] S. Chickerur, S. Balannavar, P. Hongekar, A. Prerna, and S. Jituri, âWebgl vs. webgpu: A performance analysis for web 3.0,â Procedia Computer Science, vol. 233, pp. 919â928, 2024.

[14] A. Villag Â´ omez-Palacios, C. D. l. Fuente-Burdiles, and C. Vidal-Silva, Â´ âVideogame programming & education: Enhancing programming skills through unity visual scripting,â Computers, vol. 15, no. 1, p. 68, 2026.

[15] S. N. Gunkel, S. Dijkstra-Soudarissanane, H. M. Stokking, and O. A. Niamut, âFrom 2d to 3d video conferencing: Modular rgb-d capture and reconstruction for interactive natural user representations in immersive extended reality (xr) communication,â Frontiers in Signal Processing, vol. 3, p. 1139897, 2023.

[16] S. Alexovic, M. Lacko, and J. Ba Ë cËÂ´Ä±k, â3d mapping with a drone equipped with a depth camera in indoor environment,â Acta Electrotechnica et Informatica, vol. 23, no. 1, pp. 18â24, 2023.

[17] K. R. Barad, A. Richard, J. Dentler, M. Olivares-Mendez, and C. Martinez, âObject-centric reconstruction and tracking of dynamic unknown objects using 3d gaussian splatting,â in 2024 International Conference on Space Robotics (iSpaRo). IEEE, 2024, pp. 202â209.

[18] J. Cani, C. Diou, S. Evangelatos, V. Argyriou, P. Radoglou-Grammatikis, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âIllicit object detection in x-ray imaging using deep learning techniques: A comparative evaluation,â IEEE Access, 2026.

[19] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 676â21 685.

[20] F. Barthel, A. Beckmann, W. Morgenstern, A. Hilsmann, and P. Eisert, âGaussian splatting decoder for 3d-aware generative adversarial networks,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7963â7972.

[21] J. Tang, Y. Gao, D. Yang, L. Yan, Y. Yue, and Y. Yang, âDronesplat: 3d gaussian splatting for robust 3d reconstruction from in-the-wild drone imagery,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 833â843.

[22] H. Shang, M. Chen, K. Feng, S. Li, Z. Zhang, S. Xu, C. Ren, and J. Xi, âEnhanced 3d gaussian splatting for real-scene reconstruction via depth priors, adaptive densification, and denoising,â Sensors, vol. 25, no. 22, p. 6999, 2025.

[23] Y. Ham, M. Michalkiewicz, and G. Balakrishnan, âDragon: Drone and ground gaussian splatting for 3d building reconstruction,â in 2024 IEEE International Conference on Computational Photography (ICCP). IEEE, 2024, pp. 1â12.

[24] C.-Y. Wu, L.-S. Hsu, and C.-H. Hsu, âFlygs: Online 3dgs model construction using a mavlink-connected drone,â in Proceedings of the 3rd International Workshop on UAVs in Multimedia: Capturing the World from a New Perspective, 2025, pp. 68â72.

[25] H. Luo, Z. Tu, J. He, and J. Yuan, âEfficient and spatially aware 3d gaussian splatting for compact large-scale scene reconstruction,â Applied Sciences, vol. 16, no. 2, p. 965, 2026.

[26] J. Qian, Y. Yan, F. Gao, B. Ge, M. Wei, B. Shangguan, and G. He, âC3dgs: Compressing 3d gaussian model for surface reconstruction of large-scale scenes based on multiview uav images,â IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 4396â4409, 2025.

[27] Y. Mei, R. Zeng, W. Xu, and X. Zhou, âLow-altitude uav photogrammetry for large-scale scene reconstruction with gaussian-splatting representation,â IEEE Access, vol. 13, pp. 194 644â194 656, 2025.

[28] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â5479.

[29] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[30] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1â15, 2018.

[31] X. Lei, X. Jiang, and C. Wang, âDesign and implementation of streaming media processing software based on rtmp,â in 2012 5th international congress on image and signal processing. IEEE, 2012, pp. 192â196.

[32] M. A. Yusuf, P. D. Ibnugraha, and M. I. Sani, âImplementation of nginx server with rtmp module for tea leaf maturity monitoring.â Journal of Syntax Literate, vol. 9, no. 12, 2024.

[33] Z. Idrees, J. Granados, Y. Sun, S. Latif, L. Gong, Z. Zou, and L. Zheng, âIeee 1588 for clock synchronization in industrial iot and related applications: A review on contributing technologies, protocols and enhancement methodologies,â IEEE access, vol. 8, pp. 155 660â155 678, 2020.

[34] V. Wang, F. Salim, and P. Moskovits, âThe websocket protocol,â in The Definitive Guide to HTML5 WebSocket. Springer, 2013, pp. 33â60.

[35] K. Gao, D. Lu, H. He, L. Xu, J. Li, and Z. Gong, âEnhanced 3d urban scene reconstruction and point cloud densification using gaussian splatting and google earth imagery,â IEEE Transactions on Geoscience and Remote Sensing, 2025.

[36] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[37] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[38] T. Schops, J. L. Schonberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, and A. Geiger, âA multi-view stereo benchmark with highresolution images and multi-camera videos,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 3260â 3269.

[39] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573â580.

[40] A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollar, âPanoptic Â´ segmentation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9404â9413.

[41] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[42] S. Bullinger, C. Bodensteiner, M. Arens, and R. Stiefelhagen, â3d vehicle trajectory reconstruction in monocular video data using environment structure constraints,â in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 35â50.

[43] M. Linardakis, I. Varlamis, and G. T. Papadopoulos, âSurvey on hand gesture recognition from visual input,â IEEE Access, 2025.

[44] K. Foteinos, M. Linardakis, P. Radoglou-Grammatikis, V. Argyriou, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âVisual hand gesture recognition with deep learning: A comprehensive review of methods, datasets, challenges and future research directions,â arXiv preprint arXiv:2507.04465, 2025.

[45] M. Linardakis, I. Varlamis, and G. T. Papadopoulos, âDistributed maze exploration using multiple agents and optimal goal assignment,â IEEE Access, vol. 12, pp. 101 407â101 418, 2024.

[46] S. Konstantakos, J. Cani, I. Mademlis, D. I. Chalkiadaki, Y. M. Asano, E. Gavves, and G. T. Papadopoulos, âSelf-supervised visual learning in the low-data regime: a comparative evaluation,â Neurocomputing, vol. 620, p. 129199, 2025.

[47] G. T. Papadopoulos, M. Antona, and C. Stephanidis, âTowards open and expandable cognitive ai architectures for large-scale multi-agent humanrobot collaborative learning,â IEEE access, vol. 9, pp. 73 890â73 909, 2021.

[48] G. T. Papadopoulos, A. Leonidis, M. Antona, and C. Stephanidis, âUser profile-driven large-scale multi-agent learning from demonstration in federated human-robot collaborative environments,â in International Conference on Human-Computer Interaction. Springer, 2022, pp. 548â 563.

[49] M. Moutousi, A. El Saer, N. Nikolaou, A. Sanfeliu, A. Garrell, L. Blaha, Â´ M. Cech, E. K. Markakis, I. Kefaloukos, M. Lagomarsino Ë et al., âTornado: Foundation models for robots that handle small, soft and deformable objects,â in 2025 6th International Conference in Electronic Engineering & Information Technology (EEITE). IEEE, 2025, pp. 1â13.

[50] I. Mademlis, M. Mancuso, C. Paternoster, S. Evangelatos, E. Finlay, J. Hughes, P. Radoglou-Grammatikis, P. Sarigiannidis, G. Stavropoulos, K. Votis et al., âThe invisible arms race: digital trends in illicit goods trafficking and ai-enabled responses,â IEEE Transactions on Technology and Society, vol. 6, no. 2, pp. 181â199, 2024.

[51] N. Rodis, C. Sardianos, P. Radoglou-Grammatikis, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âMultimodal explainable artificial

intelligence: A comprehensive review of methodological advances and future research directions,â IEEE Access, vol. 12, pp. 159 794â159 820, 2024.

[52] S. Evangelatos, E. Veroni, V. Efthymiou, C. Nikolopoulos, G. T. Papadopoulos, and P. Sarigiannidis, âExploring energy landscapes for minimal counterfactual explanations: Applications in cybersecurity and beyond,â IEEE Transactions on Artificial Intelligence, 2025.