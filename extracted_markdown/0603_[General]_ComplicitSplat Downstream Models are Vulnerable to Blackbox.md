# COMPLICITSPLAT: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages

Matthew Hull1, Haoyang Yang1, Pratham Mehta1, Mansi Phute 1, Aeree Cho 1, Haorang Wang1, Matthew Lau 1, Wenke Lee 1, Wilian Lunardi 2, Martin Andreoni 2, Duen Horng Chau 1,

1Georgia Tech

2Technology Innovation Institute

[matthewhull, hyang440, pratham, mphute6, aeree, haoran.wang, mattlaued01, wenke, polo]@gatech.edu,[willian.lunardi, martin.andreoni]@tii.ae

COMPLiCITSPLAT COnCealS adversarial textures visible only from certain views.

<!-- image-->  
Fig. 1: COMPLICITSPLAT conceals multiple adversarial cloaked textures in 3DGS scenes using Spherical Harmonics, causing the 3DGS representation of the car to become adversarial at different view points (red dots). For example, (A) when viewed from the top, the car appears as a suitcase, (B) âcarâ detection confidence decreases, (C) and when viewed directly from behind, displays a âstop sign.â

## Abstract

As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce COMPLICITSPLAT, the first attack that exploits standard 3DGS shading methods to create viewpointspecific camouflageâcolors and textures that change with viewing angleâto embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that COMPLICITSPLAT generalizes to successfully attack a variety of popular detectorsâboth single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.

## Introduction

3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has rapidly gained popularity in safety-critical applications due to its efficiency in novel-view synthesis from a set of static images resulting in real-time 3D rendering of complex scenes, outperforming traditional methods like Neural Radiance Fields (NeRFs) (Mildenhall et al. 2020). The advantages of 3DGS have led to growing interest in safety-critical domains such as autonomous driving (Zhou et al. 2024b; Li et al. 2024a; Huang et al. 2024), airborne navigation (Quach et al. 2024), overhead (BEV) navigation (Lei et al. 2025), and grasping (Qureshi et al. 2024; Y.Zheng et al. 2024), where rapid data generation and accurate sim2real transfer are essential.

Despite the increasing adoption of 3DGS, have the security vulnerabilities in the 3DGS scene representation been adequately considered? Currently, 3DGS scenes are widely available for download from various sources1, but associated source images are often unavailable, such as the recently announced SceneSplat-7K dataset (Li et al. 2025), which has not yet been publicly released.

Given these challenges of obtaining 3DGS without source images, what harms could an attacker cause if they tamper with the images used to construct objectsâ 3DGS representation? We expose the risk of âquiet-tamperingâ, i.e., not being able to detect that source images have been altered, resulting in COMPLICITSPLAT, a first-of-its-kind attack that conceals multiple adversarial appearances by exploiting the view-dependent nature of Spherical Harmonics (SH)âa standard technique used in real-time rendering for realistic shading, enabling an attacker to embed concealed adversarial appearances into 3DGS, each visible only from specific viewing angles. For example, Figure 1 shows how results of how COMPLICITSPLAT exploits SH shading to cause an object such as a car to appear benign from ground level and yet take on the appearance of asphalt or roadway when viewed aerially, effectively hiding from overhead surveillance systems.

With the growing use of 3DGS in autonomous driving (Zhou et al. 2024b) and robotic navigation (Quach et al. 2024; Lei et al. 2025), source images manipulated by COM-PLICITSPLAT can transform 3DGS into an unknowing accomplice, triggering misclassification and missed detections across downstream object detection tasks. Since COMPLIC-ITSPLAT perturbs only the source images used to build 3DGS without requiring access to downstream model architectures or weights, it has broad generalization potential across diverse object detection models, unlike other recent explorations in manipulation 3DGS, (Lu et al. 2024; Zeybey, Ergezer, and Nguyen 2024; Jiang et al. 2025; Hong et al. 2025), which require model access.

Our extensive experimental results demonstrate that COMPLICITSPLAT generalizes across single-stage, multistage, and transformer-based detectors in both digitallyrendered and physically-captured 3DGS scenarios, demonstrating robust adversarial effectiveness without requiring access to internal model architectures or weights.

Recent research exploration (Table 1) into manipulating 3DGS remains limited â almost no research has been open source or released publicly available code, with one exception (Lu et al. 2024); furthermore, they focus on fundamentally different goals (e.g., using extreme perturbations to drastically alter entire scene appearances or introduce severe visual artifacts) emphasizing similarity metrics between benign and perturbed scenes rather than real-world implications for downstream tasks. (Zeybey, Ergezer, and Nguyen 2024; Jiang et al. 2025; Hong et al. 2025).

In summary, our main contributions are:

â¢ Viewpoint-Specific Camouflage: First work showing that standard shading technique in 3DGS (spherical harmonics), can be exploited to conceal views for objects.

<!-- image-->  
Fig. 2: YOLOv8 detections over adversarial viewpoints attacked by COMPLICITSPLAT.

â¢ Generalizes Across Detectors: evades YOLO (v3, v5, v8, v11), FasterRCNN, and DETR detectors without requiring any access to internal model weights or architectures.

â¢ Cross-Domain Attack: Demonstrated on real-world capture process in addition to synthetic 3DGS scenes.

â¢ Reproducible Attack: We are the first work to release data and code for camouflaged 3DGS attacks, available in the Code & Data Appendix.

## Related Work

We group related research into three main areas: adversarial attacks utilizing differentiable rendering, security issues inherent in novel-view synthesis, and vulnerabilities specific to 3D Gaussian Splatting (3DGS)

## Attacks Using Differentiable Rendering

Adversarial attacks in the 2D space are well-established (Szegedy et al. 2014; Goodfellow, Shlens, and Szegedy 2015), and the corresponding vulnerabilities are extensively studied (Madry et al. 2018; Carlini et al. 2019). However, such studies are not prevalent regarding 3D spaces (Li et al. 2024b; Hull et al. 2025). Attackers have used differentiable rendering methods (Nimier-David et al. 2019; Ravi et al. 2020; Mildenhall et al. 2020; Kerbl et al. 2023) to perform adversarial gradient optimization of components in a scene, which can be used to create highly realistic scenes where perturbations are applied to geometry, texture, pose, lighting, and sensors. This results in physically plausible objects that could be transferred to the real world (Zheng et al. 2024). Even more recently, adversarial ML researchers have used NeRF and 3DGS to extend differentiable rendering attacks to novel-view synthesis, following their rising popularity in computer vision and graphics applications (Irshad et al. 2024; Zhu et al. 2024).

<table><tr><td>Property</td><td><img src="images/b01282fc29c7b6b1b995ccb123fdb01c8852f1a678feb96aa841129b62a3e48c.jpg"/></td><td>Ours</td></tr><tr><td>Viewpoint-Specific Camo</td><td></td><td></td></tr><tr><td>Generalize Across Detectors</td><td></td><td></td></tr><tr><td>Cross-Domain Attack Reproducible Attack</td><td></td><td></td></tr></table>

Table 1: Comparison of COMPLICITSPLAT with existing methods.

## Security Issues in Novel-View Synthesis Methods

Security vulnerabilities in novel-view synthesis methods using 3DGS are not extensively studied, but they share similar risks with the more thoroughly examined NeRFs, as both methods rely on training images and known camera positions. We briefly review adversarial attack literature involving NeRFs and poisoned training data, demonstrating how these vulnerabilities underscore security concerns applicable to 3DGS.

Prior work investigated novel-view exploitation in NeRFs for facial recognition evasion via template inversion attacks, highlighting practical feasibility due to minimal adversarial assumptions (no white-box model access) (Shahreza and Marcel 2023). NeRFail (Jiang et al. 2024a) applied the Iterative Gradient Signed Method (IGSM) (Kurakin, Goodfellow, and Bengio 2017) to generate adversarial pixel-space perturbations in training images, creating NeRFs capable of fooling image classifiers. Similarly, Wu et al. (2023) used Projected Gradient Descent (PGD) poisoning to induce spatial deformations in NeRF reconstructions. IPA-NeRF (Jiang et al. 2024b) introduced bi-level white-box optimization optimization with backdoor training images to embed illusory views, revealing objects from specific viewpoints but invisible elsewhere, but is limited to a appearance/disappearance, and does not allow for insertion of alternate object textures.

## Adversarial Attacks on 3D Gaussian Splatting

Limited prior work has explored adversarial vulnerabilities in 3DGS. Poison-Splat (Lu et al. 2024) introduced a computational attack targeting the split/densify stage of 3DGS training by perturbing training images to increase scene complexity, memory usage, and training time, yet did not examine impacts on downstream tasks where some autonomous systems may rely on 3DGS representations. Gaussian Splatting Under Attack (GSUA) (Zeybey, Ergezer, and Nguyen 2024) targeted only the CLIP ViT-B/16 classifier via data poisoning through segmentation and perturbation of target regions within images. GaussTrap (Hong et al. 2025) generated hidden illusory views in trained 3DGS modelsâmuch like IPA-NeRFâs âbackdoorââbut transforms the entire scene and evaluates success via imagesimilarity metrics (PSNR, SSIM, LPIPS) rather than disruptions to downstream perception. MPAM-3DGS (Jiang et al. 2025) pursues downstream task attacks by introducing multi-parametric adversarial manipulation by perturb-

One splat's outline accentuated to visualize easier to across views.

<!-- image-->  
Fig. 3: Adversarial Gaussian splats demonstrating viewdependent color changes enabled by spherical harmonic rendering. We highlight a single splat with a light border for easier tracking of color changes across views, revealing its transition from green to gray when rotating from a side view (frames AâB) to an overhead view (frames CâE).

ing Gaussian means, scales, rotations, spherical harmonic color, and alpha to attack YOLOv5 and ResNet-101. However, these parameter manipulations can produce visible artifacts (jagged splat boundaries and misalignments), making the attack more conspicuous.

However, all of the above works performed limited exploration and only one have released publicly available code, preventing comparison between methods. Furthermore, they focus primarily on extreme perturbations that drastically alter entire scene appearances or introduce severe artifacts, emphasizing similarity metrics between benign and perturbed scenes rather than real-world implications for downstream tasks.

In contrast, our method:

1. Fuses one or more adversarial appearances at the object level rather than overwriting the entire scene; concealments activate only within attacker-specified angular regions and maintain stealth by avoiding visible artifacts, unlike MPAM-3DGS.

2. Demonstrates the feasibility of how our black-box attack can be created within both digital and real-world 3DGS dataset capture process without requiring access to the model or architecture details like IPA-NeRF, PoisonSplat, GaussTrap, and MPAM-3DGS.

3. Evaluated against a wider range of object detectors than previous works and shown to be effective against multiple architectures (YOLOv3/5/8/11, Faster R-CNN, DETR) without needing to access internal model weights or architecture details.

Table 1 contrasts COMPLICITSPLAT against existing methods.

## Proposed Method: COMPLICITSPLAT

## Main Idea: Exploits Spherical Harmonics Shading

COMPLICITSPLAT leverages the view-dependent properties of 3D Gaussian Splatting (3DGS) using spherical harmonics (SH) encoding to hide adversarial content within 3D scenes (Fig. 3). Spherical harmonics (SH) form an orthonormal set of basis functions commonly used to efficiently approximate diffuse lighting and shading in computer graphics. Unlike explicit lighting modelsâsuch as Phong shading (Phong 1998), where lighting calculations depend explicitly on known scene geometry, light positions, and viewing anglesâ3DGS stores precomputed SH coefficients per Gaussian splat. This allows each splatâs appearance to change smoothly as the viewpoint shifts, without recalculating explicit light-object-camera interactions.

Typically, 3DGS is trained using SH or order $\ell = 2$ , yielding 5 basis functions and 9 coefficients per color channel, totaling 27 coefficients for RGB, effective for representing most scenarios with high accuracy (Green 2003). Reducing â results in a limit on color estimation, e.g., $\ell \ = \ 1$ uses only four coefficients per channel (12 total), These coefficients parameterize a continuous directional color function that is fitted during training; SH coefficients are optimized across multiple camera views, effectively capturing realistic color variations such as reflections and intricate lighting conditions. Notably, this process gives an adversary considerable capabilities to embed multiple adversarial appearances within a scene for some target object and viewing angles.

We exploit the view-dependent nature of SH by causing the model to learn a desired adversarial appearance only visible from specific angles through replacement of training images with adversarial images, enabling sophisticated concealment. For example, a car can be designed with an adversarial appearance from a top view while maintaining benign appearances from all other angles (Fig. 1, Fig. 2). Walking 360 degrees around such a vehicle on the ground appears completely normal, as the top of the car viewed from ground level shows no indication of the hidden adversarial content.

## Threat Model

We first present the threat model used in our 3DGS attack.

Attackerâs goals: The adversary aims to embed concealed adversarial content into the reconstructed 3D scene such that it is only visible from specific viewpoints in the rendered output. The objective is to ensure that this content is hidden from general inspection but reliably appears under targeted viewing conditions, thereby manipulating downstream applications or human observers.

Attackerâs knowledge: The attacker is assumed to have knowledge of the overall 3DGS pipeline and its training process, including the use of Gaussian primitives and SH coefficients for scene representation. However, the attacker does not have access to internal scene parameters, intermediate representations, or the final trained 3DGS scene or any models used in downstream tasks.

Attackerâs capabilities: The adversary is restricted to manipulating the training data only. This includes altering or injecting images into the training set but does not extend to modifying the 3DGS algorithm, its loss functions, or the internal optimization process. The attacker cannot directly access or modify the scene parameters during or after training.

Attackerâs strategy: The attacker crafts or selects training images such that, when the 3DGS model is trained on this data, the resulting volumetric scene encodes the adversarial content in a way that it is only revealed from pre-determined or rare viewpoints. The attack leverages the modelâs learning process to âhideâ content within the volumetric representation, exploiting the view-dependent rendering of the 3DGS pipeline to trigger the adversarial effect only under specific conditions.

## Problem Formulation & Algorithm

We formulate our COMPLICITSPLAT attack as follows. Let $\mathbfcal { D } = \{ ( x _ { i } , c _ { i } ) \} _ { i = } ^ { N }$ be the benign dataset, where each image $x _ { i } ~ \in ~ X$ is associated with a camera pose $c _ { i } ~ \in ~ C$ The attacker selects a set of $n$ targeted viewpoint regions $R ^ { * } ~ = ~ \{ R _ { 1 } ^ { * } , R _ { 2 } ^ { * } , . . . , R _ { n } ^ { * } \}$ , each defined relative to corresponding reference viewpoints $c _ { \mathrm { r e f } , i }$ and angular thresholds $\delta _ { i }$ for each appearance they wish to conceal.

$$
R _ { i } ^ { * } = \{ c \in C : \angle ( c , c _ { \mathrm { r e f } , i } ) \leq \delta _ { i } \} , \quad i = 1 , \ldots , n ,\tag{1}
$$

where $\angle ( c , c _ { \mathrm { r e f } , i } )$ denotes the angular distance between viewing directions (camera optical axes) of poses c and $c _ { \mathrm { r e f } , i } .$ For each region $R _ { i } ^ { * }$ , the attacker generates adversarial images $\tilde { x } _ { i } ^ { ( j ) }$ corresponding to a distinct concealed appearance $j ,$ modifying the appearance of the target object while maintaining the sceneâs visual realism and structural consistency. The attacker then replaces each original image $x _ { i }$ with its adversarial counterpart corresponding to the region it belongs to, forming the attacked dataset:

$$
\mathcal { D } ^ { \prime } = \{ ( A ( x _ { i } , c _ { i } ) , c _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{2}
$$

where

$$
A ( x , c ) = \left\{ \begin{array} { l l } { \tilde { x } ^ { ( j ) } , } & { \mathrm { i f } c \in R _ { j } ^ { * } , j = 1 , \ldots , n , } \\ { x , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

The attack algorithm iterates exactly once through all N images in the dataset D. For each image, checking membership of the camera position $c _ { i }$ within the n targeted viewpoint regions $R _ { i } ^ { * }$ takes $O ( N )$ . Similarly, the space complexity is also $O ( N )$ , which involves storing the original dataset $O ( N )$ , adversarial images (up to n), and the modified dataset $O ( N )$

Training the 3DGS model on $\mathcal { D } ^ { \prime }$ ensures that from nontargeted viewpoints $c \notin R ^ { * }$ (where $R ^ { * } = \textstyle \bigcup _ { i = 1 } ^ { n } R _ { i } ^ { * } )$ , the target object retains its benign appearance. Conversely, viewpoints within each targeted region $R _ { i } ^ { * }$ reveal the corresponding embedded adversarial appearance $\tilde { x } ^ { ( i ) }$ . Multiple adversarial appearances can thus be smoothly concealed and independently revealed as the viewpoint transitions through attacker-defined angular regions, subject only to the capacity of the sceneâs spherical harmonic (SH) representation. Training 3DGS scenes with adversarially manipulated images causes competing appearances across viewpoints, effectively âpushing and pullingâ image similarity metrics (SSIM) during optimization, resulting in slower convergenceâapproximately 2Ã slower compared to benign scenes (0.5 minutes vs. 15 seconds to reach SSIM â¥ 0.95 on NVIDIA RTX 4090 GPU). However, in practice, scenes are typically trained to 30K iterations, and our scenes reach high SSIM (â¥ 0.93) and appear highly realistic

## Evaluation

In this section, we evaluate the effectiveness and robustness of our adversarial attack against multiple popular object detection models trained on the COCO dataset. We focus on two safety-critical adversarial scenarios relevant to autonomous navigation: an overhead vehicle scenario, where a car is disguised as part of the roadway or grass when viewed from above, and a ground-level stop sign scenario, where concealed markings become visible only from specific viewing angles. Vehicles and stop signs are common, safety-critical targets, making these scenarios ideal for investigating vulnerabilities in real-world object detection systems (Quach et al. 2024; Lei et al. 2025).

<!-- image-->  
Fig. 4: Camera layout for data collection in both scenarios. Left: Overhead vehicle scenario with cameras distributed across a hemisphere. Right: Ground-level stop sign scenario with cameras covering a 90-degree arc.

We assess the attacks against YOLO (v3, v5, v8, v11), Faster R-CNN, and DETR object detectors. These specific models were selected to cover a broad spectrum of detection architectures: YOLO versions represent single-stage detectors known for their lower computational demands and realtime performance (Cao et al. 2023); Faster RCNN was chosen for its multi-stage detection process and higher accuracy at the cost of increased complexity (Leng et al. 2024); and DETR (Carion et al. 2020) was included as a representative of transformer-based detection architectures, which offer a fundamentally different approach compared to traditional CNN-based models.

## Experimental Setup

In our overhead vehicle scenario, we trained the 3DGS scene using 200 images rendered with Blender, covering a full 360-degree hemisphere around vehicles positioned within a realistic city-street environment, following the capture setup used by Mip-NeRF (Barron et al. 2022). We use 30K iterations training each 3DGS (Kerbl et al. 2023). We evaluate two camouflage textures, a âroadâ texture and âgrassâ textures based upon their potential to hide objects overhead and plausibility of occurrence in street environment views using 3 color variants (gray, red, and blue) of a car. For testing, we constructed 5 additional hemispheres of increasing size and chose 160 test overhead views at random points along the overhead region of the hemispheres, thereby evaluating the attack on unseen viewpoints (Fig. 4-left).

For the ground-based scenario involving the stop sign, we rendered 144 images using Blender within the same citystreet context, capturing a 90-degree field of view. This setup provided complete visibility of one face of the stop sign, showing it from the left edge to a full front-facing view. For camouflage, we chose âclockâ and âsoccerâ ball textures, visible only when rotating to the full front-facing view, but concealed when viewed from side views greater than 30 degrees from the front (Fig. 4-right).

<!-- image-->  
Fig. 5: Benign and adversarial views of real-world physical attack on a model car.

Attacking 3DGS Captured Digitally We generate adversarially perturbed 3DGS scenes following the COMPLICIT-SPLAT attack formulation introduced earlier. Given a benign dataset $\mathcal { D } = \{ ( x _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N }$ , each image $x _ { i } \in X$ is associated with a camera pose $c _ { i } \in C$ . We select specific targeted viewpoint regions $\grave { R } ^ { * }$ in (Eq. 1).

In the overhead vehicle scenario, we exemplify this twostep rendering process. Initially, we render the original benign scene, generating images xi capturing the standard appearance of the targeted vehicle from camera poses $c _ { i } .$ . Subsequently, we identify attacker-specified viewpoints within a region $\mathbf { \bar { \boldsymbol { R } } } _ { i } ^ { * }$ (e.g., overhead angles) from which a concealed adversarial appearance will be visible. For these viewpoints, we alter the objectâs appearance by applying an adversarial camouflage texture (e.g., road pavement), rendering corresponding adversarial images $\tilde { x } _ { i } ^ { ( j ) }$ . The original images $x _ { i }$ associated with these targeted viewpoints are replaced by their adversarial counterparts $\tilde { x } _ { i } ^ { ( j ) }$ , forming the attacked dataset $\mathcal { D } ^ { \prime } \left( \mathrm { E q . } 2 \right)$

Training the 3DGS model on $\mathcal { D } ^ { \prime }$ maintains the structural consistency of the scene, preserving object positions and orientations. From non-targeted viewpoints $c \notin R ^ { * }$ (where $R ^ { * } = \textstyle \bigcup _ { i = 1 } ^ { n } R _ { i } ^ { * } )$ , the scene remains benign. Conversely, the adversarial appearances embedded in each region $\bar { R _ { i } ^ { * } }$ become visible as the viewpoint transitions into these attackerspecified regions, revealing concealed adversarial content tailored specifically to each region.

Attacking 3DGS Captured in Real-World To validate our adversarial approach in realistic 3DGS workflows, we extended our synthetic data method to real-world captures using accessible, low-cost tools. For the overhead vehicle scenario, we prepared two identical physical car models (Fig. 5): one benign (blue) and one adversarial (painted with road camouflage) and scanned them individually using PolyCam, a mobile app that captures 3D models using photogrammetry techniques and producing textured 3D meshes compatible with Blender. In Blender, we precisely aligned and rendered both models from identical camera posesâground-level views for the benign car, overhead views for the camouflaged car. The combined image set trained a 3DGS scene that preserved structural consistency, resulting in a vehicle that appeared benign at ground-level but revealed concealed adversarial appearance from overhead attacker-defined viewpoints.

A  
<!-- image-->

B  
<!-- image-->

<!-- image-->  
Fig. 6: Drop in AP@0.5 IoU for camouflage attacks on cars (road/grass) and stop signs (clock/soccer) across all detectors. Lower AP indicates more effective camouflage.

## Evaluation Metrics:

We employ several metrics to comprehensively evaluate the performance of our adversarial attacks across different object detection models:

â¢ Attack Success Rate (ASR): The percentage of images where the targeted object is misclassified or not detected under adversarial conditions compared to benign conditions, as used in other adversarial attacks on object detection (Chen et al. 2019).

â¢ Average Precision (AP): Measures the precision of the model at different recall levels. We report AP at IoU thresholds of 0.5 (AP@0.5), as done in previous work on camouflage adversarial attacks (Suryanto et al. 2023; Zhou et al. 2024a).

## Results and Analysis

We report experiments to answer the following questions:

Q1. Viewpoint-Specific Camouflage: How reliably can COMPLICITSPLAT disguise targeted objects (e.g., vehicles, stop signs) from attacker-chosen viewing angles, causing misclassification or missed-detections for object detectors (Fig. 4)?

Q2. Detector Generalization: Does COMPLICITSPLAT consistently evade detection across multiple object detection architectures, including lightweight singlestage (YOLO), multi-stage (Faster RCNN), and transformer-based (DETR) models?

Q3. Attack on Real-World Capture: To what extent do adversarial attacks maintain effectiveness when applied to 3DGS capture of real physical-world objects?

## Q1. Viewpoint-Specific Camouflage

The effectiveness of COMPLICITSPLAT in disguising targeted objects (e.g., vehicles as roads, stop sign as clocks) is measured by the attack success rate (ASR) on viewing angles in the test set of viewing angles. Using the overhead based car as an example, ASR is the fraction of images where a car detected under benign conditions is not detected as a car under adversarial conditions. Table 2 shows ASR for the overhead-based (car) and the ground-based (stop-sign), respectively.

Table 2: Combined Attack Success Rate (ASR) for all adversarial camouflages used with Stop Sign and Cars Â by model.
<table><tr><td></td><td colspan="2">â¢ Clock</td><td colspan="2">Soccer</td><td colspan="2">Grass</td><td colspan="2">Road</td></tr><tr><td>Model</td><td>Suc./Tot. ASR (%) Suc./Tot. ASR Suc./Tot. ASR Suc./Tot.</td><td></td><td></td><td></td><td></td><td></td><td></td><td>ASR</td></tr><tr><td>YOLOv3</td><td>59 / 123</td><td>47.97</td><td>24 / 123</td><td>19.51</td><td>54 /126</td><td>42.86</td><td>14 /126</td><td>11.11</td></tr><tr><td>YOLOv5</td><td>58 / 120</td><td>48.33</td><td>48 / 120</td><td>40.00</td><td>77 /119</td><td>64.71</td><td>37 /119</td><td>31.09</td></tr><tr><td>YOLOv8</td><td>68 /117</td><td>58.12</td><td>56/117</td><td>47.86</td><td>49 /50</td><td>98.00</td><td>50 / 50</td><td>100.00</td></tr><tr><td>YOLOv11</td><td>72 /114</td><td>63.16</td><td>59/114</td><td>51.75</td><td>53 /92</td><td>57.61</td><td>75 /92</td><td>81.52</td></tr><tr><td>FRCNN</td><td>88 / 105</td><td>83.81</td><td>49 / 105</td><td>46.67</td><td>57/65</td><td>87.69</td><td>58/65</td><td>89.23</td></tr><tr><td>DETR</td><td>69 / 128</td><td>53.91</td><td>49 / 128</td><td>38.28</td><td>43/57</td><td>75.44</td><td>24/57</td><td>42.11</td></tr></table>

Overall, we observe that COMPLICITSPLAT achieves high attack success rates across more recently released detectors, ranging from 50% (DETR) - 91% (YOLOv8). Interestingly, YOLOv3/v5 show higher robustness â some previous evaluation suggest that earlier YOLO models (v5) can outperform later models (v8) (KÄ±lÄ±cÂ¸kaya, TasÂ¸yurek, and Â¨ Ozt Â¨ urk 2023) in vehicle detection. Â¨

## Q2. Detector Generalization

Next, we assess whether COMPLICITSPLAT consistently evades detection across diverse object detection architectures by measuring change in AP@0.5 under âroadâ and âgrassâ camouflage on overhead cars and âclockâ and âsoccerâ camouflage on stop signs (Figure 6). All detectors exhibit substantial AP@0.5 reductions across scenarios. For cars, grass camouflage generally causes larger performance degradation than road camouflage (e.g., YOLOv11 drops 0.41 under grass vs. 0.32 under road, while DETR drops 0.35 vs. 0.13). For stop signs, camouflage is even more effective: clock textures produce drops of up to 0.52 (YOLOv11) and 0.47 (DETR), and soccer textures also reduce AP@0.5 by 0.34â0.43 across models.

## Q3. Attack on Real-World Capture

Finally, we test the robustness of adversarial attacks when applied to realistic physical-world capture of the model car scenario as described in our Experimental Setup. The realworld ASR results in Table 3 show that adversarial appearances generated via 3DGS remain effective when applied to real-world images, although with varied efficacy across models. While YOLOv8 and DETR maintain moderate success rates (58.82% and 68.75%, respectively), YOLOv3 and YOLOv11 exhibit lower transferability (18.60% and 37.86%). Notably, FasterRCNN remains highly susceptible (98.51%), showing that multi-stage detectors can be vulnerable in physical deployments. These findings suggest that 3DGS-based attacks can generalize beyond digital renders, but model architecture plays an important role in real-world robustness.

Table 3: Attack success rate (ASR) for real-world car images
<table><tr><td>Model</td><td>Object</td><td>Successful / Total</td><td>ASR (%)</td></tr><tr><td>YOLOv3</td><td>a</td><td>24 / 129</td><td>18.60</td></tr><tr><td>YOLOv5</td><td></td><td>35 / 115</td><td>30.43</td></tr><tr><td>YOLOv11</td><td></td><td>39 / 103</td><td>37.86</td></tr><tr><td>YOLOv8</td><td></td><td>40 / 68</td><td>58.82</td></tr><tr><td>DETR</td><td></td><td>33 / 48</td><td>68.75</td></tr><tr><td>FRCNN</td><td></td><td>66 / 67</td><td>98.51</td></tr></table>

## Ablations

In our experimental setup, we aimed to explore the impact of two factors on the effectiveness of our adversarial camouflage attacks, leading us to conduct ablation studies on the following:

â¢ Number of Spherical Harmonics Coefficients Spherical harmonics (SH) coefficients determine the complexity of the camouflage appearance. Higher SH orders allow for more detailed estimation of object colors during training, potentially allowing better capture of camouflage patterns and increasing the effectiveness of the attack.

â¢ Camera Distances from Target Object The distance of the camera from the target object influences the visibility and effectiveness of the camouflage.

SH Order Ablation. To ablate the number of spherical harmonics coefficients, we varied the SH order used in the 3DGS training process, using orders â = 0, 1, 2 and then evaluated the Average Precision (AP) at IoU threshold 0.5 for the Â car under âgrassâ adversarial camouflage conditions, using the same camera poses as in the main experiments (Fig. 4-left).

In Table 4, we observe that lowering the SH order does not consistently reduce attack success, suggesting that restricting SH expressivity alone may not reliably mitigate adversarial camouflage effectiveness.

Table 4: AP@0.5 for Â car with SH ablations.
<table><tr><td>SH Order</td><td>YOLOv3</td><td>YOLOv5</td><td>YOLOv8</td><td>YOLOv11</td><td>FRCNN</td><td>DETR</td></tr><tr><td>=2</td><td>0.485</td><td>0.267</td><td>0.020</td><td>0.277</td><td>0.050</td><td>0.109</td></tr><tr><td> = 1</td><td>0.495</td><td>0.238</td><td>0.020</td><td>0.317</td><td>0.040</td><td>0.129</td></tr><tr><td> = 0</td><td>0.475</td><td>0.287</td><td>0.030</td><td>0.297</td><td>0.050</td><td>0.198</td></tr></table>

Camera Distance Ablation. For ablation of camera distance on adversarial camouflage effectiveness, we took each of the 5 partial hemispheres (Fig. 4-left) and evaluated them separately, measure AP and AR at IoU threshold 0.5 for the Â car under âgrassâ adversarial camouflage conditions and averaging across all detectors, presenting results by average altitude (in meters) above the top of the car.

We observe a trend that as camera altitude increases, the effectiveness of the camouflage improves (Table 5). AP and AR degradation becomes more severe at higher altitudes, with the largest drops occurring between 20-30 meters, suggesting that elevated vantage points amplify the adversarial impact of 3DGS-based appearances.

Table 5: Average AP@0.5 / AR@0.5 across all detectors for benign and grass appearance at each altitude scenario.
<table><tr><td>Altitude (m)</td><td>Benign AP/AR</td><td>Adv AP/AR</td><td>âAP/ âAR</td></tr><tr><td>12</td><td>0.218 / 0.214</td><td>0.033 / 0.031</td><td>-0.185 / -0.183</td></tr><tr><td>16</td><td>0.223 / 0.217</td><td>0.022 / 0.018</td><td>-0.201 /-0.199</td></tr><tr><td>20</td><td>0.387 / 0.379</td><td>0.058 / 0.051</td><td>-0.329 / -0.328</td></tr><tr><td>24</td><td>0.436 / 0.437</td><td>0.073 / 0.071</td><td>-0.363 / -0.366</td></tr><tr><td>30</td><td>0.436 / 0.436</td><td>0.073 / 0.071</td><td>-0.363 / -0.365</td></tr></table>

## Mitigation Strategies

Training Data Scrutiny. Could increased scrutiny and careful vetting of training datasets mitigate the effectiveness of adversarial attacks? Approaches could include inspection for identifying unusual or suspicious textures and employing automated anomaly detection tools capable of flagging potential adversarial inputs based on visual or statistical irregularities.

Limiting Spherical Harmonics Coefficients. Would limiting the complexity of spherical harmonics (SH) used in the representation effectively reduce adversarial risks? Table 4 shows mixed results on whether reducing SH order would sufficiently constrain an attacker but could shows potential for transformer-based models, such as DETR.

## Conclusion

We presented COMPLICITSPLAT, the first black-box attack targeting 3D Gaussian Splats (3DGS) by exploiting spherical harmonics to embed adversarial appearances within scenes, precisely controlling object visibility at attackerdesignated viewpoints (Section ). The attack reliably generalizes across diverse object detectors without requiring internal model details, remaining effective in both digital and real-world domains, and is fully reproducible via our opensource implementation available in the Code & Data Appendix.

## References

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. ArXiv:2111.12077 [cs].

Cao, Z.; Kooistra, L.; Wang, W.; Guo, L.; and Valente, J. 2023. Real-Time Object Detection Based on UAV Remote Sensing: A Systematic Literature Review. Drones, 7(10).

Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-End Object Detection with Transformers. In Vedaldi, A.; Bischof, H.; Brox, T.; and Frahm, J.-M., eds., Computer Vision â ECCV 2020, 213â 229. Cham: Springer International Publishing.

Carlini, N.; Athalye, A.; Papernot, N.; Brendel, W.; Rauber, J.; Tsipras, D.; Goodfellow, I.; Madry, A.; and Kurakin, A. 2019. On Evaluating Adversarial Robustness. arXiv:1902.06705 [cs, stat]. ArXiv: 1902.06705.

Chen, S.-T.; Cornelius, C.; Martin, J.; and Chau, D. H. 2019. ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector. arXiv:1804.05810 [cs, stat], 11051: 52â68. ArXiv: 1804.05810.

Goodfellow, I. J.; Shlens, J.; and Szegedy, C. 2015. Explaining and Harnessing Adversarial Examples. arXiv:1412.6572 [cs, stat]. ArXiv: 1412.6572.

Green, R. 2003. Spherical Harmonic Lighting: The Gritty Details. In Game Developerâs Conference. San Jose, CA, USA.

Hong, J.; Chen, S.; Sun, S.; Yu, H.; Fang, H.; Tan, Y.; Chen, B.; Qi, S.; and Li, J. 2025. GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion. ArXiv:2504.20829 [cs].

Huang, N.; Wei, X.; Zheng, W.; An, P.; Lu, M.; Zhan, W.; Tomizuka, M.; Keutzer, K.; and Zhang, S. 2024. S3 Gaussian: Self-Supervised Street Gaussians for Autonomous Driving. ArXiv:2405.20323 [cs].

Hull, M.; Wang, H.; Lau, M.; Helbling, A.; Phute, M.; Zhang, C.; Kira, Z.; Lunardi, W.; Andreoni, M.; Lee, W.; and Chau, D. H. 2025. RenderBender: A Survey on Adversarial Attacks Using Differentiable Rendering. In IJCAI.

Irshad, M. Z.; Comi, M.; Lin, Y.-C.; Heppert, N.; Valada, A.; Ambrus, R.; Kira, Z.; and Tremblay, J. 2024. Neural Fields in Robotics: A Survey. eprint: 2410.20220.

Jiang, W.; Zhang, H.; Wang, W.; Guo, Z.; Zhang, T.; and Wang, H. 2025. MPAM-3DGS: Multi-Parametric Adversarial Manipulation for 3D Gaussian Splatting. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1â5.

Jiang, W.; Zhang, H.; Wang, X.; Guo, Z.; and Wang, H. 2024a. NeRFail: Neural Radiance Fields-Based Multiview Adversarial Attack. Proceedings of the AAAI Conference on Artificial Intelligence.

Jiang, W.; Zhang, H.; Zhao, S.; Guo, Z.; and Wang, H. 2024b. IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields. In ECAI 2024, 513â520. IOS Press.

Kerbl, B.; Kopanas, G.; Leimkuehler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4): 1â14.

Kurakin, A.; Goodfellow, I. J.; and Bengio, S. 2017. Adversarial Machine Learning at Scale. In International Conference on Learning Representations.

KÄ±lÄ±cÂ¸kaya, F. N.; TasÂ¸yurek, M.; and Â¨ Ozt Â¨ urk, C. 2023. Perfor- Â¨ mance evaluation of YOLOv5 and YOLOv8 models in car detection. Imaging and Radiation Research, 6(2).

Lei, X.; Wang, M.; Zhou, W.; and Li, H. 2025. GaussNav: Gaussian Splatting for Visual Navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5): 4108â4121.

Leng, J.; Ye, Y.; Mo, M.; Gao, C.; Gan, J.; Xiao, B.; and Gao, X. 2024. Recent Advances for Aerial Object Detection: A Survey. ACM Computing Surveys, 56(12): 1â36.

Li, H.; Li, J.; Zhang, D.; Wu, C.; Shi, J.; Zhao, C.; Feng, H.; Ding, E.; Wang, J.; and Han, J. 2024a. VDG: Vision-Only Dynamic Gaussian for Driving Simulation.

Li, Y.; Ma, Q.; Yang, R.; Li, H.; Ma, M.; Ren, B.; Popovic, N.; Sebe, N.; Konukoglu, E.; Gevers, T.; Gool, L. V.; Oswald, M. R.; and Paudel, D. P. 2025. SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining.

Li, Y.; Xie, B.; Guo, S.; Yang, Y.; and Xiao, B. 2024b. A Survey of Robustness and Safety of 2D and 3D Deep Learning Models against Adversarial Attacks. ACM CSur., 56(6).

Lu, J.; Zhang, Y.; Shen, Q.; Wang, X.; and Yan, S. 2024. Poison-splat: Computation Cost Attack on 3D Gaussian Splatting. ArXiv:2410.08190 [cs].

Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2018. Towards Deep Learning Models Resistant to Adversarial Attacks. In ICLR.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Computer Vision - ECCV 2020, volume 12346, 405â421.

Cham: Springer International Publishing. Series Title: Lecture Notes in Computer Science.

Nimier-David, M.; Vicini, D.; Zeltner, T.; and Jakob, W. 2019. Mitsuba 2: a retargetable forward and inverse renderer. ACM Transactions on Graphics, 38(6): 1â17.

Phong, B. T. 1998. Illumination for computer generated pictures. In Seminal graphics: pioneering efforts that shaped the field, 95â101. Communications.

Quach, A.; Chahine, M.; Amini, A.; Hasani, R.; and Rus, D. 2024. Gaussian Splatting to Real World Flight Navigation Transfer with Liquid Networks. ArXiv:2406.15149 [cs].

Qureshi, M. N.; Garg, S.; Yandun, F.; Held, D.; Kantor, G.; and Silwal, A. 2024. SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting. In CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data.

Ravi, N.; Reizenstein, J.; Novotny, D.; Gordon, T.; Lo, W.- Y.; Johnson, J.; and Gkioxari, G. 2020. Accelerating 3D Deep Learning with PyTorch3D. ArXiv:2007.08501 [cs].

Shahreza, H.; and Marcel, S. 2023. Comprehensive Vulnerability Evaluation of Face Recognition Systems to Template Inversion Attacks via 3D Face Reconstruction. TPAMI, 45(12): 14248â14265.

Suryanto, N.; Kim, Y.; Larasati, H. T.; Kang, H.; Le, T.-T.- H.; Hong, Y.; Yang, H.; Oh, S.-Y.; and Kim, H. 2023. AC-TIVE: Towards Highly Transferable 3D Physical Camouflage for Universal and Robust Vehicle Evasion. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 4282â4291. Paris, France: IEEE. ISBN 979-8- 3503-0718-4.

Szegedy, C.; Zaremba, W.; Sutskever, I.; Bruna, J.; Erhan, D.; Goodfellow, I.; and Fergus, R. 2014. Intriguing properties of neural networks. arXiv:1312.6199 [cs]. ArXiv: 1312.6199.

Wu, Y.; Feng, B. Y.; and Huang, H. 2023. Shielding the Unseen: Privacy Protection through Poisoning NeRF with Spatial Deformation. ArXiv:2310.03125 [cs].

Y.Zheng; Chen, X.; Zheng, Y.; Gu, S.; Yang, R.; Jin, B.; Li, P.; Zhong, C.; Wang, Z.; Liu, L.; Yang, C.; Wang, D.; Chen, Z.; Long, X.; and Wang, M. 2024. GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping. IEEE Robotics and Automation Letters, 9(9): 7827â7834.

Zeybey, A.; Ergezer, M.; and Nguyen, T. 2024. Gaussian Splatting Under Attack: Investigating Adversarial Noise in 3D Objects. In Neurips Safe Generative AI Workshop 2024.

Zheng, J.; Lin, C.; Sun, J.; Zhao, Z.; Li, Q.; and Shen, C. 2024. Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 24452â24461.

Zhou, J.; Lyu, L.; He, D.; and LI, Y. 2024a. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation. In Forty-first International Conference on Machine Learning.

Zhou, X.; Lin, Z.; Shan, X.; Wang, Y.; Sun, D.; and Yang, M. 2024b. DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes. In (CVPR).

Zhu, S.; Wang, G.; Kong, X.; Kong, D.; and Wang, H. 2024. 3D Gaussian Splatting in Robotics: A Survey. ArXiv:2410.12262 [cs].

Table 6: AP@0.5 / AR@0.5 for benign vs. road-camouflage appearance across detectors for the overhead-based vehicle scenario.
<table><tr><td>Model</td><td>Object</td><td>Appearance</td><td>AP@0.5</td><td>âAP</td><td>AR@0.5</td><td>âAR</td></tr><tr><td>YOLOv3</td><td></td><td>Benign</td><td>0.792</td><td></td><td>0.797</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.733</td><td>-0.059</td><td>0.738</td><td>-0.059</td></tr><tr><td></td><td></td><td>Grass</td><td>0.485</td><td>-0.307</td><td>0.481</td><td>-0.316</td></tr><tr><td></td><td>a</td><td>Benign</td><td>0.743</td><td></td><td>0.747</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.673</td><td>-0.070</td><td>0.677</td><td>-0.070</td></tr><tr><td></td><td></td><td>Grass</td><td>0.485</td><td>-0.258</td><td>0.481</td><td>-0.266</td></tr><tr><td></td><td></td><td>Benign</td><td>0.703</td><td></td><td>0.708</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.515</td><td>-0.188</td><td>0.519</td><td>-0.189</td></tr><tr><td></td><td></td><td>Grass</td><td>0.485</td><td>-0.218</td><td>0.481</td><td>-0.227</td></tr><tr><td>YOLOv5</td><td></td><td>Benign</td><td>0.743</td><td></td><td>0.744</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.545</td><td>-0.198</td><td>0.547</td><td>-0.197</td></tr><tr><td></td><td></td><td>Grass</td><td>0.267</td><td>-0.476</td><td>0.266</td><td>-0.478</td></tr><tr><td></td><td></td><td>Benign</td><td>0.733</td><td></td><td>0.736</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.614</td><td>-0.119</td><td>0.613</td><td>-0.123</td></tr><tr><td></td><td></td><td>Grass</td><td>0.307 0.703</td><td>-0.426</td><td>0.303 0.708</td><td>-0.433</td></tr><tr><td></td><td></td><td>Benign Road</td><td>0.465</td><td>-0.238</td><td>0.468</td><td>-0.240</td></tr><tr><td></td><td></td><td>Grass</td><td>0.257</td><td>-0.446</td><td>0.256</td><td>-0.452</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YOLOv8</td><td></td><td>Benign</td><td>0.317</td><td></td><td>0.316</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.010 0.020</td><td>-0.307</td><td>0.007</td><td>-0.309</td></tr><tr><td></td><td></td><td>Grass</td><td>0.465</td><td>-0.297</td><td>0.013</td><td>-0.303</td></tr><tr><td></td><td></td><td>Benign</td><td></td><td></td><td>0.469</td><td>-0.366</td></tr><tr><td></td><td></td><td>Road</td><td>0.109 0.188</td><td>-0.356</td><td>0.103</td><td></td></tr><tr><td></td><td></td><td>Grass</td><td>0.307</td><td>-0.277</td><td>0.190</td><td>-0.279</td></tr><tr><td></td><td></td><td>Benign</td><td>0.010</td><td></td><td>0.302</td><td></td></tr><tr><td></td><td></td><td>Road Grass</td><td>0.079</td><td>-0.297 -0.228</td><td>0.006 0.079</td><td>-0.296 -0.223</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YOLOv11</td><td></td><td>Benign</td><td>0.564</td><td></td><td>0.568</td><td></td></tr><tr><td></td><td></td><td>Road</td><td>0.109 0.277</td><td>-0.455</td><td>0.106</td><td>-0.462</td></tr><tr><td></td><td></td><td>Grass</td><td>0.663</td><td>-0.287</td><td>0.275</td><td>-0.293</td></tr><tr><td></td><td>a</td><td>Benign</td><td>0.287</td><td>-0.376</td><td>0.663 0.283</td><td>-0.380</td></tr><tr><td></td><td></td><td>Road</td><td>0.455</td><td>-0.208</td><td>0.450</td><td>-0.213</td></tr><tr><td></td><td></td><td>Grass</td><td>0.465</td><td></td><td>0.463</td><td></td></tr><tr><td></td><td></td><td>Benign Road</td><td>0.069</td><td>-0.396</td><td>0.068</td><td>-0.395</td></tr><tr><td></td><td></td><td>Grass</td><td>0.327</td><td>-0.138</td><td>0.323</td><td>-0.140</td></tr><tr><td>FasterRCNN</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Benign</td><td>0.396 0.069</td><td>-0.327</td><td>0.396 0.067</td><td>-0.329</td></tr><tr><td></td><td></td><td>Road</td><td>0.050</td><td>-0.346</td><td>0.049</td><td>-0.347</td></tr><tr><td></td><td></td><td>Grass</td><td>0.406</td><td></td><td>0.409</td><td></td></tr><tr><td></td><td></td><td>Benign</td><td>0.079</td><td>-0.327</td><td>0.073</td><td>-0.336</td></tr><tr><td></td><td></td><td>Road</td><td>0.178</td><td>-0.228</td><td>0.171</td><td>-0.238</td></tr><tr><td></td><td></td><td>Grass</td><td>0.317</td><td></td><td>0.317</td><td></td></tr><tr><td></td><td></td><td>Benign</td><td>0.109</td><td>-0.208</td><td>0.104</td><td>-0.213</td></tr><tr><td></td><td></td><td>Road Grass</td><td>0.089</td><td>-0.228</td><td>0.085</td><td>-0.232</td></tr><tr><td>DETR</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Benign</td><td>0.347</td><td>-0.070</td><td>0.348</td><td>-0.074</td></tr><tr><td></td><td></td><td>Road</td><td>0.277 0.109</td><td>-0.238</td><td>0.274 0.104</td><td>-0.244</td></tr><tr><td></td><td></td><td>Grass</td><td>0.436</td><td></td><td>0.439</td><td></td></tr><tr><td></td><td></td><td>Benign</td><td>0.228</td><td>-0.208</td><td>0.226</td><td>-0.213</td></tr><tr><td></td><td></td><td>Road</td><td></td><td>-0.238</td><td>0.195</td><td>-0.244</td></tr><tr><td></td><td></td><td>Grass Benign</td><td>0.198 0.347</td><td></td><td>0.341</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>-0.109</td><td>0.232</td><td>-0.109</td></tr><tr><td></td><td></td><td>Road</td><td>0.238</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Grass</td><td>0.178</td><td>-0.169</td><td>0.177</td><td>-0.164</td></tr></table>

Table 7: AP@0.5 / AR@0.5 for benign vs. adversarial roadcamouflage on a single object (Stop Sign) across detectors
<table><tr><td>Model</td><td>Object</td><td>Appearance</td><td>AP@0.5</td><td>âAP</td><td>AR@0.5</td><td>âAR</td></tr><tr><td>YOLOv3</td><td>â¢</td><td>Benign</td><td>0.851</td><td></td><td>0.854</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.446</td><td>-0.405</td><td>0.444</td><td>-0.410</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.683</td><td>-0.168</td><td>0.688</td><td>-0.166</td></tr><tr><td>YOLOv5</td><td>â¢</td><td>Benign</td><td>0.832</td><td></td><td>0.833</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.436</td><td>-0.396</td><td>0.431</td><td>-0.402</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.505</td><td>-0.327</td><td>0.500</td><td>-0.333</td></tr><tr><td>YOLOv8</td><td>â¢</td><td>Benign</td><td>0.851</td><td></td><td>0.854</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.356</td><td>-0.495</td><td>0.358</td><td>-0.496</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.436</td><td>-0.415</td><td>0.436</td><td>-0.418</td></tr><tr><td>YOLOv11</td><td>â¢</td><td>Benign</td><td>0.822</td><td></td><td>0.820</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.307</td><td>-0.515</td><td>0.302</td><td>-0.518</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.396</td><td>-0.426</td><td>0.396</td><td>-0.424</td></tr><tr><td>FasterRCNN</td><td>â¢</td><td>Benign</td><td>0.723</td><td></td><td>0.729</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.119</td><td>-0.604</td><td>0.118</td><td>-0.611</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.386</td><td>-0.337</td><td>0.389</td><td>-0.340</td></tr><tr><td>DETR</td><td>â¢</td><td>Benign</td><td>0.881</td><td></td><td>0.889</td><td></td></tr><tr><td></td><td></td><td>Clock</td><td>0.416</td><td>-0.465</td><td>0.417</td><td>-0.472</td></tr><tr><td></td><td></td><td>Soccer</td><td>0.545</td><td>-0.336</td><td>0.549</td><td>-0.340</td></tr></table>

Table 8: AP@0.5 / AR@0.5 for benign vs. adversarial on overhead blue-car scenario across detectors.
<table><tr><td>Model</td><td>Object</td><td>Condition</td><td>AP@0.5</td><td>âAP</td><td>AR@0.5</td><td>âAR</td></tr><tr><td>YOLOv3</td><td>A</td><td>Benign</td><td>0.822</td><td></td><td>0.822</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.703</td><td>-0.119</td><td>0.707</td><td>-0.115</td></tr><tr><td>YOLOv5</td><td>A</td><td>Benign</td><td>0.713</td><td></td><td>0.714</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.535</td><td>-0.178</td><td>0.539</td><td>-0.175</td></tr><tr><td>YOLOv8</td><td></td><td>Benign</td><td>0.436</td><td></td><td>0.430</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.287</td><td>-0.149</td><td>0.285</td><td>-0.145</td></tr><tr><td>YOLOv11</td><td>A</td><td>Benign</td><td>0.634</td><td></td><td>0.640</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.436</td><td>-0.198</td><td>0.438</td><td>-0.202</td></tr><tr><td>Detectron2</td><td>A</td><td>Benign</td><td>0.406</td><td></td><td>0.409</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.010</td><td>-0.396</td><td>0.006</td><td>-0.403</td></tr><tr><td>DETR</td><td></td><td>Benign</td><td>0.297</td><td></td><td>0.293</td><td></td></tr><tr><td></td><td></td><td>Adv</td><td>0.139</td><td>-0.158</td><td>0.134</td><td>-0.159</td></tr></table>