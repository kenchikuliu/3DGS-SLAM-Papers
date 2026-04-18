# Interactive Augmented Reality-enabled Outdoor Scene Visualization For Enhanced Real-time Disaster Response

Dimitrios Apostolakisâ, Georgios Angelidisââ , Vasileios Argyriouâ¡, Panagiotis SarigiannidisÂ§, Georgios Th. Papadopoulosââ 

âDepartment of Informatics and Telematics, Harokopio University of Athens, Athens, Greece â Archimedes, Athena Research Center, Athens, Greece

â¡Department of Networks and Digital Media, Kingston University, London, United Kingdom Â§Department of Electrical and Computer Engineering, University of Western Macedonia, Kozani, Greece Emails: {it2022004, gangelidis, g.th.papadopoulos}@hua.gr, Vasileios.Argyriou@kingston.ac.uk, psarigiannidis@uowm.gr

AbstractâA user-centered AR interface for disaster response is presented in this work that uses 3D Gaussian Splatting (3DGS) to visualize detailed scene reconstructions, while maintaining situational awareness and keeping cognitive load low. The interface relies on a lightweight interaction approach, combining Worldin-Miniature (WIM) navigation with semantic Points of Interest (POIs) that can be filtered as needed, and it is supported by an architecture designed to stream updates as reconstructions evolve. User feedback from a preliminary evaluation indicates that this design is easy to use and supports real-time coordination, with participants highlighting the value of interaction and POIs for fast decision-making in context. Thorough user-centric performance evaluation demonstrates strong usability of the developed interface and high acceptance ratios.

Index TermsâAugmented Reality (AR), 3D Gaussian Splatting (3DGS), Disaster Response, Situational Awareness, Human-Computer Interaction (HCI), World-in-Miniature (WIM), Realtime Scene Reconstruction.

## I. INTRODUCTION

Effective emergency response and Search and Rescue (SAR) missions often occur under extreme time pressure and in challenging conditions [1]. In these contexts, First Responders (FRs) must maintain strong Situational Awareness (SA) [2] to support both sound decision-making and operational safety [3]â[5]. Traditionally, commanders and field teams depend on 2D maps, voice-based communication, and contrary data feeds to construct a Common Operational Picture (COP) [6]. Yet, converting this two-dimensional information into an accurate understanding of the three-dimensional real-world places a heavy cognitive burden on users by forcing them to perform mental transformations and assumptions about geometry and spatial relationships. In time-critical scenarios, these added demands can slow situation assessment and contribute to avoidable navigation or coordination errors [7], [8].

<!-- image-->  
Fig. 1. Snapshot of the developed AR-enabled 3D visualization of the disaster outdoor scene, including the main supported interface functionalities.

To overcome these limitations, Augmented Reality (AR) has gained traction as an effective means to connect digital information with the physical environment [9]â[12]. By overlaying key data directly onto the userâs visual field, AR interfaces promote ârecognition rather than recall,â in line with one of Nielsenâs core usability heuristics [13]. Nevertheless, AR is not merely a visualization comfort but a safety-critical interface, meaning that users may rely on it for time-sensitive judgments (e.g., navigation and hazard avoidance), so UI errors or latency can translate into operational mistakes and safety risks. Performance therefore matters beyond raw throughput [14]. Latency and unstable frame timing can cause spatial hints to lag or drift relative to head motion, undermining distance and alignment judgments and complicating safe coordination. Yet, depicting complex disaster environments requires rendering enormous amounts of geometric detail (e.g., rubble, fires, structural degradation), on mobile or wearable hardware handle inefficiently.

Recently, 3DGS [15] has gained attention as an efficient scene-reconstruction technique, delivering photorealistic rendering with speeds that exceed those of traditional Neural Radiance Fields (NeRFs) [16]â[19]. Even though 3DGS is capable of delivering the visual accuracy required for achieving precise damage assessment, its efficiency as a credible AR solution for real-time applications is not yet carefully investigated. In particular, there is still limited interpretation regarding the scalability of 3DGS on resource-constrained devices. Moreover, disaster environments are naturally dynamic: hazards can emerge or spread, access routes may become blocked or rerouted, and damaged structures can further decay over time. Consequently, a static reconstruction may become outdated within minutes and risk delivering an inaccurate picture of the scene, underscoring the need for timely, realtime updates to maintain reliable situational awareness.

In this work, a user-centered AR-based Human-Computer Interface (HCI) is introduced, tailored to disaster response operations. This approach leverages 3DGS to achieve precisionlevel visualization of evolving scenes while following usability best practices to reduce cognitive load on FRs. The main contributions of this paper are as follows:

â¢ A system architecture that connects an AR client to a lightweight backend to support real-time dynamic updates of 3DGS scenes without interrupting user interaction.

â¢ An AR interface design that prioritizes SA through notintrusive visual elements, applicable to both PC-AR and standalone AR hardware.

â¢ A comprehensive performance evaluation across mobile, standalone and desktop AR configurations, highlighting hardware constraints for large-scale disaster scenes, complemented by a pilot user study on usability and coordination support.

The rest of this paper is organized as follows: Section II reviews the related works on 3D scene reconstruction and the use of VR-based visualization in emergency response situations. Section III presents the methodology of the proposed system. Section IV details the experimental setup and reports the performance together with findings from the pilot user study. Section VI discusses the main limitations and drawbacks encountered, and outlines directions for future work. Finally, Section VII concludes the paper and summarizes the key findings.

## II. RELATED WORK

The potential of immersive technologies to aid emergency decision-making was identified early, with seminal studies showing their usefulness in rescue operations as early as the mid-1990s [12]. However, over the past decade, the field has experienced a paradigm shift. Propelled by progress in photogrammetry and computer vision [20]â[22], research has advanced from basic simulations to highly accurate Digital Twins and AI-supported SA [9], [23].

Nevertheless, much of the literature that currently exists examines the use of immersive technologies from the perspective of training. Hancko et al. [11] present a review that describes the use of immersive technologies to train firefighters, highlighting the possibilities that the technology has to offer, including the potential to improve safety, repeatability, and scalability of training, as well as the benefits that are applicable to the fire service, such as improved SA, decision-making, and teamwork, and the estimation of cost efficiency. Other literature, such as that presented by Lovreglio et al. [24], who compared the use of VR to conduct training on the use of fire extinguishers with video-based training, found that participants who used the VR reported better knowledge and confidence, while Gong et al. [25] developed an earthquake training system that places the participant in a virtual ecosystem, enabling the researcher to observe and evaluate the reaction to the situation.

In parallel, several studies show how integrating highquality geospatial data with interactive 3D environments can support crisis management workflows. Tully et al. [26] introduced the Project Vision Support framework, which integrates LiDAR, OpenStreetMap, and Ordnance Survey data within a game engine to generate hybrid, high-resolution 3D maps for crisis response. Similarly, Velev et al. [27] discuss how UAVs paired with AR can empower near real-time disaster mapping, allowing responders to identify critical locations and assess structural damage from a safer distance.

Generally, these works point to an assembled ecosystem where data-rich 3D reconstruction, immersive interfaces, and rapid mapping pipelines boost each other, supporting both operational decision-making during incidents and realistic, repeatable training for emergency crew. Taken together, these studies highlight the value of immersive 3D environments for situational awareness and decision support. However, most reported systems emphasize training scenarios or static/postdisaster reconstructions. The proposed system shifts the role of VR from training to operations. VR becomes a real-time workspace for coordination, where the 3D scene is updated dynamically as new information arrives.

## III. METHODOLOGY

A back-end service is used to reconstruct a 3D scene from UAV data as it is collected. The backend processes the incoming measurements and generates a â.plyâ file that represents the reconstructed environment. This file is then streamed to the front-end via WebSockets, sanctioning near real-time updates. More details about the back-end development are described here [28].

On the front-end side, the system uses an interactive engine designed for building immersive experiences. A lightweight user interface allows the user to interact with the scene (e.g., inspection and navigation in it). The reconstructed 3D environment is rendered directly from the streamed â.plyâ data and can be visualized not only on a standard display, but also through VR headsets for an immersive, first-person view of the scene.

## A. AR Interface and Interaction Design

The proposed interaction framework is designed with the overall goal of optimizing the balance between SA and cognitive load in high-pressure environments. Grounded on Cognitive Load Theory [29] and established HCI guidelines [13], the system follows an adaptive, minimalist approach that emphasizes task-critical information, while limiting visual clutter and redundant interaction steps. Consistent with the ârecognition rather than recallâ heuristic, essential information is presented through in-view contextual cues and explicit highlighting of salient POIs, thereby reducing unnecessary mental effort. This design choice is supported by recent work showing that the âdetail-in-contextâ visualization strategy significantly enhances the operatorâs ability to form an accurate mental representation of remote environments [30]. In addition, the system operates as a cognitive artifact to foster a shared mental model (common ground) among distributed teams. This functionality is vital for counteracting performance losses under stress in unfamiliar settings [31] and for supporting the interpretation of complex disaster scenarios through highfidelity simulation [23].

1) Scene Interaction: Because disaster scenes often cover large areas, physical movement alone is insufficient for a fast inspection. Rather than depending exclusively on teleportation, a WIM [32] manipulation technique is integrated. Users manipulate the map directly via hand controllers. With these controllers, they can perform standard affine transformations:

â¢ Scaling: Scaling is achieved through bimanual controller interaction, allowing users to smoothly transition between a global and a local view of the environment. This enables fluid switching between a high-level overview of the site and a close-up examination of specific rubble.

â¢ Rotation and Translation: Users can rotate and reposition the map to reveal otherwise occluded regions without having to physically move around the room, which is especially advantageous in tight operational settings.

Finally, the interface includes a reset function that allows users to restore all scene transformations (e.g., translation, rotation, and scale) to their default state, for quick recovery from unintended manipulations during time-critical operations.

2) Visualization of POIs: Disaster environments are intrinsically disordered. To support interpretation, the system introduces semantic layers. Through UI, users can toggle individual hazard classes (e.g., âFireâ, âSmokeâ, âDebrisâ, etc.) on or off. In doing so, commanders can focus on mission-critical information without being distracted by irrelevant visual elements. At the same time, this approach allows task-oriented data exploration, while maintaining a critical understanding of the overall situation [30].

3) Spatial awareness via Passthrough: To keep operators safe and to make teamwork easier on site, the system uses mixed-reality video passthrough so that users can still see what is happening around them while viewing digital overlays. Unlike fully immersive VR, passthrough helps users feel less cut off and makes it easier to talk and coordinate with nearby colleagues. This lets an operator quickly switch from reading digital information to taking real-world action, an important capability in fast-changing emergency situations. For configurations that lack passthrough capabilities, limitations of which are discussed in Section VI, the interface reverts to a VR environment. To balance the loss of physical context, the system provides locomotion controls [33] through the controllers for detailed exploration, so the user could navigate âinsideâ the 3D scene and inspect the terrain with the same level of immersion as an on-site responder.

## IV. EXPERIMENTS AND EVALUATION

To verify the effectiveness of the systemâs interaction, navigation, and visualization capabilities, a preliminary evaluation was conducted involving 12 participants, a sample size of 10 Â± 2, selected according to [34]. This evaluation focused on assessing how well users could interact with the 3D reconstructed environment, navigate through the disaster scene, and perceive the visual quality of the reconstructed data.

Also, a performance experiment was executed across mobile AR, desktop AR and standalone AR headsets, to check the rendering computational load and performance of a 3D postdisaster reconstructed scene under different hardware constraints. Frame rate (FPS) served as the main evaluation metric, since maintaining high and stable FPS is essential for user comfort and conscious interaction, especially in immersive VR/AR settings [35], [36]. Furthermore, frame time was measured, which is simply how long it takes to draw a single frame on screen. Lower and more consistent frame times make motion look smoother and the system feel more responsive.

## A. Experimental Setup

For our purpose, because the custom rendering pipeline for 3DGS was designed in Unity [37], [38], Unity 2022.3 is used as the front-end, while UI/UX are reused components from MRTK3 [39], as they align with the design guidelines discussed in Section III-A. For the benchmarking process, a high-fidelity 3D reconstruction based on Gaussian Splatting is tested. This scene consists of 1,144,277 Gaussian splats, being depicted in Fig 2, capturing complex geometries, such as debris, concrete structures, and scattered hazards. This dataset was chosen to stress-test the sorting algorithms, as the scene visually resembles a disaster-like environment.

<!-- image-->  
Fig. 2. Snapshot of the 1.114 million debug points of the 3DGS sample used for the evaluation study from the Unity Editor

Three hardware setups are evaluated:

â¢ Mobile AR used a Samsung S24 FE (Exynos 2400e, Samsung Xclipse 940, 8 GB RAM), capturing a lightweight, handheld AR scenario.

â¢ Standalone VR used a Meta Quest 3 (Adreno 740 GPU), representing an untethered, fully immersive VR experience.

â¢ PC-AR relied on a desktop workstation with an Intel Core i7 and an NVIDIA RTX 3080 Ti (12 GB VRAM), streaming to the Meta Quest 3 via Oculus Link, which reflects to the proposed base-station architecture.

Mobile and standalone VR deployments rely on Vulkan, due to API support for compute operations. Desktop deployment utilizes DirectX 12 to maximize GPU throughput.

## B. Performance Experiment Results

Table I summarizes the rendering performance and runtime characteristics across the three tested devices. In terms of frame rate, the desktop configuration maintained 72 FPS, while both mobile and the standalone VR headset remained below 10 FPS (â¤7 and â¤9 FPS, respectively).

TABLE I  
DEVICES AND FPS COMPARISONS
<table><tr><td>Device</td><td>GPU</td><td>FPS (Avg)</td></tr><tr><td>Mobile</td><td>Samsung Xclipse 940</td><td>7</td></tr><tr><td>Meta (AR)</td><td>Adreno 740</td><td>9</td></tr><tr><td>Desktop</td><td>NVIDIA GeForce RTX 3080 Ti</td><td>72</td></tr></table>

The memory and timing results in Table II highlight clear differences in how each device uses resources. On the PC, the application shows the highest overall RAM footprint, with 1485 MB reserved and 937 MB allocated, while still achieving a relatively low frame time of 13.8 ms. In contrast, Mobile and standalone AR, use less memory overall, with 387 MB reserved and 229 MB allocated on mobile and 504 MB reserved and 411 MB allocated on standalone, but they run with much higher frame times at 144.4 ms and 111.3 ms, respectively. Mono memory follows the same pattern, remaining much higher on PC at 527 MB and dropping to 8 MB on mobile and 5 MB on standalone.

TABLE II  
MEMORY USAGE AND FRAME TIME (MS) ACROSS DEVICES
<table><tr><td>Device</td><td>Reserved</td><td>Allocated</td><td>Mono</td><td>Frame Time (ms)</td></tr><tr><td>Desktop</td><td>1485 MB</td><td>937 MB</td><td>527 MB</td><td>13.8</td></tr><tr><td>Mobile</td><td>387 MB</td><td>229 MB</td><td>8 MB</td><td>144.4</td></tr><tr><td>Meta (AR)</td><td>504 MB</td><td>411 MB</td><td>5 MB</td><td>111.3</td></tr></table>

## V. USER EVALUATION RESULTS

The user evaluation results in Table III show that the mean of user perception ease of map interaction was 4.42, menu accessibility was 4.36, and usefulness for coordination was 4.55, while the standard deviations was 0.82, 0.84, and 0.52, respectively.

TABLE III  
USER EVALUATION RESULTS (N=12) ON A 5-POINTS LIKERT SCALE
<table><tr><td>Evaluation Criterion</td><td>Mean Score (Âµ)</td><td>SD (Ï)</td></tr><tr><td>Ease of Map Interaction</td><td>4.42</td><td>0.82</td></tr><tr><td>Menu Accessibility</td><td>4.36</td><td>0.84</td></tr><tr><td>Usefulness for Coordination</td><td>4.55</td><td>0.52</td></tr></table>

## VI. DISCUSSION, LIMITATIONS AND FUTURE WORK

The results of the user evaluation study showed that the user interface was highly usable, with positive ratings for usability and perceived support for coordination, as all the mean ratings were above 4 out of 5, and the highest agreement was obtained for the usefulness of the interface in coordination. Additionally, based on the performance experiments, the PC-AR configuration is currently the most suitable option for visualization, achieving a stable 72 FPS with a 13.8 ms frame time. In this setup, a desktop workstation at the ground station absorbs the computational cost required to render the 3DGS scene, while the view is streamed to the headset. However, according to Metaâs documentation, passthrough for applications connected to a PC via Oculus Link is intended for development use [40] and is available only when running through the Unity Editor, not in a standalone Windows build. Prior work [37], [38] reports stable 72 FPS up to approximately 400k splats, which indicates that a standalone headset deployment can also be feasible under appropriate scene complexity, while additionally supporting pass-through. Finally, because splats are rendered as an image-based representation rather than physical geometry, interaction is handled using a simplified approach: a collider is placed around the boundary of the map and manipulation is performed based on that proxy collider.

In future work, based on the received feedback, the system will be extended with map annotation capabilities, providing the coordinator the ability to mark and comment on areas of interest directly within the reconstructed scene. Multi-user support will be explored, allowing multiple coordinators to simultaneously view and collaborate in the same environment. Beyond headset-based interaction, interaction patterns that do not require wearable equipment will be investigated. Additionally,

POIs will be made interactive so that coordinators can select them and view additional useful information. Moreover, the system will be evaluated with FRs to validate its effectiveness in simulated operational scenarios. Furthermore, the system will be connected with real-time AI-enabled event detectors [41]â[43], so as to facilitate scenarios involving Human-Robot Interaction [44]â[46] and broader security response incidents [47], while also supporting the necessary eXplainable Artificial Intelligence (XAI) pipelines [48], [49].

## VII. CONCLUSIONS

In this paper, a user-centered AR/VR interface for disaster management was presented, that uses 3DGS to visualize high-accuracy reconstructions while prioritizing SA and low cognitive load. By combining a lightweight interaction design (WIM-based navigation and semantic POIs toggles) with an architecture that supports streaming updates of reconstructed scenes, the system is positioned as an operational workspace rather than a training tool. This aims to help coordinators and field teams to build and to maintain a shared understanding of rapidly changing environments.

A preliminary evaluation with 12 participants provides early evidence that the interaction design is effective and easy to use. Participants rated 3D map interaction and menu access highly, reported that POIs and POI filtering support faster, more focused decision-making. They also perceived the system as useful for real-time coordination, suggesting that the interface can support shared understanding in time-critical scenarios.

The performance study highlights a key practical outcome: large scale 3DGS scenes remain challenging for mobile and standalone headsets when rendered locally. In contrast, for scenes that contain massive amounts of splats, PC-AR sustained a stable 72 FPS, while mobile AR and standalone AR stayed below 10 FPS. This gap indicates that, for high-detail disaster reconstruction, a base-station approach (rendering on a workstation, streaming to a headset) is currently the most reliable option for responsive interaction and comfortable frame timing. At the same time, the observed limitations, especially around pass-through availability in PC-AR configuration and the simplified interaction afforded by splatbased representations, show that system-level constraints are as important as raw reconstruction quality, when deploying AR for safety-critical use.

Future work will extend the system towards real operational deployment by supporting multi-user collaboration, in-scene annotations for coordination, and richer POI interaction to surface additional contextual information.

## REFERENCES

[1] J. Cani, P. Koletsis, K. Foteinos, I. Kefaloukos, L. Argyriou, M. Falelakis, I. Del Pino, A. Santamaria-Navarro, M. Cech, O. Severa Ë et al., âTriffid: Autonomous robotic aid for increasing first responders efficiency,â in 2025 6th International Conference in Electronic Engineering & Information Technology (EEITE). IEEE, 2025, pp. 1â9.

[2] M. R. Endsley, D. J. Garland et al., âTheoretical underpinnings of situation awareness: A critical review,â Situation awareness analysis and measurement, vol. 1, no. 1, pp. 3â21, 2000.

[3] W. Nasar, R. Da Silva Torres, O. E. Gundersen, and A. T. Karlsen, âThe use of decision support in search and rescue: A systematic literature review,â ISPRS International Journal of Geo-Information, vol. 12, no. 5, 2023. [Online]. Available: https://www.mdpi.com/2220-9964/12/5/182

[4] K. Steen-Tveit and J. Radianti, âAnalysis of common operational picture and situational awareness during multiple emergency response scenarios.â in ISCRAM, 2019.

[5] R. R. Lutz, âSafe-ar: Reducing risk while augmenting reality,â in 2018 IEEE 29th International Symposium on Software Reliability Engineering (ISSRE), 2018, pp. 70â75.

[6] A. Agrawal and J. Cleland-Huang, âRescuear: Augmented reality supported collaboration for uav driven emergency response systems,â arXiv preprint arXiv:2110.00180, 2021.

[7] M. Vuckovic, J. Schmidt, T. Ortner, and D. Cornel, âCombining 2d and 3d visualization with visual analytics in the environmental domain,â Information, vol. 13, no. 1, 2022. [Online]. Available: https://www.mdpi.com/2078-2489/13/1/7

[8] S. Sharma, S. T. Bodempudi, D. Scribner, J. Grynovicki, and P. Grazaitis, âEmergency response using hololens for building evacuation,â in International Conference on Human-Computer Interaction. Springer, 2019, pp. 299â311.

[9] S. Khanal, U. S. Medasetti, M. Mashal, B. Savage, and R. Khadka, âVirtual and augmented reality in the disaster management technology: a literature review of the past 11 years,â Frontiers in Virtual Reality, vol. 3, p. 843195, 2022.

[10] M. Chmielewski, K. Sapiejewski, and M. Sobolewski, âApplication of augmented reality, mobile devices, and sensors for a combat entity quantitative assessment supporting decisions and situational awareness development,â Applied Sciences, vol. 9, no. 21, p. 4577, 2019.

[11] D. Hancko, A. Majlingova, and D. KacËÂ´Ä±kova, âIntegrating virtual reality, Â´ augmented reality, mixed reality, extended reality, and simulation-based systems into fire and rescue service training: Current practices and future directions,â Fire, vol. 8, no. 6, 2025. [Online]. Available: https://www.mdpi.com/2571-6255/8/6/228

[12] G. E. Beroggi, L. Waisel, and W. A. Wallace, âEmploying virtual reality to support decision making in emergency management,â Safety Science, vol. 20, no. 1, pp. 79â88, 1995, the International Emergency Management and Engineering Society. [Online]. Available: https://www.sciencedirect.com/science/article/pii/092575359400068E

[13] J. Nielsen, âEnhancing the explanatory power of usability heuristics,â in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, ser. CHI â94. New York, NY, USA: Association for Computing Machinery, 1994, p. 152â158. [Online]. Available: https://doi.org/10.1145/191666.191729

[14] M. Mirbabaie and J. Fromm, âReducing the cognitive load of decisionmakers in emergency management through augmented reality,â 2019.

[15] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[16] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[17] S. Liu, M. Yang, T. Xing, and R. Yang, âA survey of 3d reconstruction: The evolution from multi-view geometry to nerf and 3dgs,â Sensors, vol. 25, no. 18, p. 5748, 2025.

[18] C. Blanchard, L. Gupta, and S. Nanisetty, âAnalyzing 3d gaussian splatting and neural radiance fields: A comparative study on complex scenes and sparse views,â cs. tornto. edu, 2023.

[19] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, âCompact 3d gaussian representation for radiance field,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 21 719â21 728.

[20] S. Konstantakos, J. Cani, I. Mademlis, D. I. Chalkiadaki, Y. M. Asano, E. Gavves, and G. T. Papadopoulos, âSelf-supervised visual learning in the low-data regime: a comparative evaluation,â Neurocomputing, vol. 620, p. 129199, 2025.

[21] P. Alimisis, I. Mademlis, P. Radoglou-Grammatikis, P. Sarigiannidis, and G. T. Papadopoulos, âAdvances in diffusion models for image data augmentation: A review of methods, models, evaluation metrics and future research directions,â Artificial Intelligence Review, vol. 58, no. 4, p. 112, 2025.

[22] J. Cani, C. Diou, S. Evangelatos, V. Argyriou, P. Radoglou-Grammatikis, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âIllicit object detection in x-ray imaging using deep learning techniques: A comparative evaluation,â IEEE Access, 2026.

[23] Y. Zhu and N. Li, âVirtual and augmented reality technologies for emergency management in the built environments: A state-of-the-art review,â Journal of safety science and resilience, vol. 2, no. 1, pp. 1â10, 2021.

[24] R. Lovreglio, X. Duan, A. Rahouti, R. Phipps, and D. Nilsson, âComparing the effectiveness of fire extinguisher virtual reality and video training,â Virtual Reality, vol. 25, no. 1, pp. 133â145, 2021.

[25] X. GONG, Y. LIU, Y. JIAO, B. WANG, J. ZHOU, and H. YU, âA novel earthquake education system based on virtual reality,â IEICE Transactions on Information and Systems, vol. E98.D, no. 12, pp. 2242â 2249, 2015.

[26] D. Tully, A. El Rhalibi, C. Carter, and S. Sudirman, âHybrid 3d rendering of large map data for crisis management,â ISPRS International Journal of Geo-Information, vol. 4, no. 3, pp. 1033â1054, 2015.

[27] D. Velev, P. Zlateva, L. Steshina, and I. Petukhov, âChallenges of using drones and virtual/augmented reality for disaster risk management,â The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. 42, pp. 437â440, 2019.

[28] C. Maikos, G. Angelidis, and G. T. Papadopoulos, âLarge-scale photorealistic outdoor 3d scene reconstruction from uav imagery using gaussian splatting techniques,â 2026. [Online]. Available: https://arxiv.org/abs/2602.20342

[29] N. Hollender, C. Hofmann, M. Deneke, and B. Schmitz, âIntegrating cognitive load theory and concepts of humanâcomputer interaction,â Computers in Human Behavior, vol. 26, no. 6, pp. 1278â1288, 2010, online Interactivity: Role of Technology in Behavior Change. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0747563210001718

[30] R. Bakzadeh, K. M. Joao, V. Androulakis, H. Khaniani, S. Shao, M. Hassanalian, and P. Roghanchi, âEnhancing emergency response: The critical role of interface design in mining emergency robots,â Robotics, vol. 14, no. 11, 2025. [Online]. Available: https://www.mdpi. com/2218-6581/14/11/148

[31] M. Migliorini, L. Licata, and D. Strumendo, âVirtual and augmented reality for disaster risk reduction,â in 1st Croatian Conference on Earthquake Engineering, 2021, p. 8.

[32] R. Stoakley, M. J. Conway, and R. Pausch, âVirtual reality on a wim: interactive worlds in miniature,â in Proceedings of the SIGCHI conference on Human factors in computing systems, 1995, pp. 265â272.

[33] R. Pausch, T. Burnette, D. Brockway, and M. E. Weiblen, âNavigation and locomotion in virtual worlds via flight into hand-held miniatures,â in Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, 1995, pp. 399â400.

[34] W. Hwang and G. Salvendy, âNumber of people required for usability evaluation: the 10Â±2 rule,â Commun. ACM, vol. 53, no. 5, p. 130â133, May 2010. [Online]. Available: https://doi.org/10.1145/ 1735223.1735255

[35] A. Geris, B. Cukurbasi, M. Kilinc, and O. Teke, âBalancing performance and comfort in virtual reality: A study of fps, latency, and batch values,â Software: Practice and Experience, vol. 54, no. 12, pp. 2336â2348, 2024.

[36] D. J. Zielinski, H. M. Rao, M. A. Sommer, and R. Kopper, âExploring

the effects of image persistence in low frame rate virtual environments,â in 2015 IEEE Virtual Reality (VR), 2015, pp. 19â26.

[37] C. Kleinbeck, H. Schieber, K. Engel, R. Gutjahr, and D. Roth, âMultilayer gaussian splatting for immersive anatomy visualization,â IEEE Transactions on Visualization and Computer Graphics, vol. 31, no. 5, pp. 2353â2363, 2025.

[38] A. Pranckevicius24, âUnity gaussian splatting,â https://github.com/ Ë aras-p/UnityGaussianSplatting, 2024.

[39] Microsoft, âMixed reality toolkit 3 (mrtk3),â https://github.com/ MixedRealityToolkit/MixedRealityToolkit-Unity, 2023, version 3.0, accessed: December 2025.

[40] Meta Platforms, Inc., âPassthrough over link,â https: //developers.meta.com/horizon/documentation/native/android/ mobile-passthrough-over-link/, accessed: December 2025.

[41] M. Linardakis, I. Varlamis, and G. T. Papadopoulos, âSurvey on hand gesture recognition from visual input,â IEEE Access, 2025.

[42] K. Foteinos, M. Linardakis, P. Radoglou-Grammatikis, V. Argyriou, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âVisual hand gesture recognition with deep learning: A comprehensive review of methods, datasets, challenges and future research directions,â arXiv preprint arXiv:2507.04465, 2025.

[43] M. Linardakis, I. Varlamis, and G. T. Papadopoulos, âDistributed maze exploration using multiple agents and optimal goal assignment,â IEEE Access, vol. 12, pp. 101 407â101 418, 2024.

[44] G. T. Papadopoulos, M. Antona, and C. Stephanidis, âTowards open and expandable cognitive ai architectures for large-scale multi-agent humanrobot collaborative learning,â IEEE access, vol. 9, pp. 73 890â73 909, 2021.

[45] G. T. Papadopoulos, A. Leonidis, M. Antona, and C. Stephanidis, âUser profile-driven large-scale multi-agent learning from demonstration in federated human-robot collaborative environments,â in International Conference on Human-Computer Interaction. Springer, 2022, pp. 548â 563.

[46] M. Moutousi, A. El Saer, N. Nikolaou, A. Sanfeliu, A. Garrell, L. Blaha, Â´ M. Cech, E. K. Markakis, I. Kefaloukos, M. Lagomarsino Ë et al., âTornado: Foundation models for robots that handle small, soft and deformable objects,â in 2025 6th International Conference in Electronic Engineering & Information Technology (EEITE). IEEE, 2025, pp. 1â13.

[47] I. Mademlis, M. Mancuso, C. Paternoster, S. Evangelatos, E. Finlay, J. Hughes, P. Radoglou-Grammatikis, P. Sarigiannidis, G. Stavropoulos, K. Votis et al., âThe invisible arms race: digital trends in illicit goods trafficking and ai-enabled responses,â IEEE Transactions on Technology and Society, vol. 6, no. 2, pp. 181â199, 2024.

[48] N. Rodis, C. Sardianos, P. Radoglou-Grammatikis, P. Sarigiannidis, I. Varlamis, and G. T. Papadopoulos, âMultimodal explainable artificial intelligence: A comprehensive review of methodological advances and future research directions,â IEEe Access, vol. 12, pp. 159 794â159 820, 2024.

[49] S. Evangelatos, E. Veroni, V. Efthymiou, C. Nikolopoulos, G. T. Papadopoulos, and P. Sarigiannidis, âExploring energy landscapes for minimal counterfactual explanations: Applications in cybersecurity and beyond,â IEEE Transactions on Artificial Intelligence, 2025.