<!-- page 1 -->
RePaintGS: Reference-Guided Gaussian Splatting
for Realistic and View-Consistent 3D Scene Inpainting
Ji Hyun Seoa,b, Byounhyun Yooa,c,∗, Gerard Jounghyun Kimb,∗
aIntelligence and Interaction Research Center, Korea Institute of Science and Technology, 5 Hwarang-ro, Seongbuk-gu, Seoul, 02792, Republic of Korea
bDepartment of Computer Science and Engineering, Korea University, 145 Anam-ro, Seongbuk-gu, Seoul, 02841, Republic of Korea
cAI-Robotics, KIST School, Korea National University of Science and Technology, 5 Hwarangro14-gil, Seongbuk-gu, Seoul, 02792, Republic of Korea
Abstract
Radiance field methods, such as Neural Radiance Field or 3D Gaussian Splatting, have emerged as seminal 3D representations for
synthesizing realistic novel views. For practical applications, there is ongoing research on flexible scene editing techniques, among
which object removal is a representative task. However, removing objects exposes occluded regions, often leading to unnatural
appearances. Thus, studies have employed image inpainting techniques to replace such regions with plausible content—a task
referred to as 3D scene inpainting. However, image inpainting methods produce one of many plausible completions for each
view, leading to inconsistencies between viewpoints. A widely adopted approach leverages perceptual cues to blend inpainted
views smoothly. However, it is prone to detail loss and can fail when there are perceptual inconsistencies across views. In this
paper, we propose a novel 3D scene inpainting method that reliably produces realistic and perceptually consistent results even for
complex scenes by leveraging a reference view. Given the inpainted reference view, we estimate the inpainting similarity of the
other views to adjust their contribution in constructing an accurate geometry tailored to the reference. This geometry is then used to
warp the reference inpainting to other views as pseudo-ground truth, guiding the optimization to match the reference appearance.
Comparative evaluation studies have shown that our approach improves both the geometric fidelity and appearance consistency of
inpainted scenes. For more details, please visit our project page.
Keywords: 3D gaussian splatting, 3D reconstruction, 3D scene inpainting, radiance field
1. Introduction
Radiance field methods, such as Neural Radiance Field
(NeRF) [1, 2, 3] and 3D Gaussian Splatting (3DGS) [4, 5, 6],
have emerged as leading and indispensable techniques for 3D
representation due to their ability to synthesize highly realis-
tic novel views. These technologies are increasingly replac-
ing traditional 3D representation methods in applications such
as Digital Twins, Virtual/Augmented Reality (VR/AR), and re-
mote collaboration. To enable more versatile usage, demand
for editable radiance fields [7] has been increasing, prompting
active research efforts to support flexible scene modifications.
Among various editing tools, such as relighting [8, 9, 10] and
style transfer [11, 12], object removal is considered one of the
essential features for complete scene editing, allowing users to
eliminate unnecessary elements and thus support more flexible
scene rearrangement. However, removing object exposes previ-
ously occluded regions, often leading to an unnatural or incom-
plete appearance [13]. Reconstructing these regions is particu-
larly challenging when they are not visible from any viewpoint.
To address this, techniques that generate plausible 3D regions
that blend naturally with the surrounding scene are needed.
∗Co-corresponding authors.
Email addresses: jihyun.seo@wrl.onl (Ji Hyun Seo), yoo@byoo.org
(Byounhyun Yoo), gjkim@korea.ac.kr (Gerard Jounghyun Kim)
Nevertheless, current 3D-aware generative models [14, 15, 16]
are typically optimized for limited types of training data and
simplified scene structures, making it difficult to generalize to
complex and diverse real-world environments.
Accordingly,
studies
applied
image
inpainting
tech-
niques [17, 18, 19] to the radiance field method to reconstruct
hidden regions of the 3D scene, a task referred to as 3D scene
inpainting. However, because image inpainting methods are
applied independently per view and do not consider the 3D
consistency in structure and texture, they produce inconsisten-
cies that can lead to distortions and hallucinations in radiance
fields.
Therefore, 3D scene inpainting research focuses on
attempts to blend image inpainting results into a plausible and
coherent 3D scene. A notable work, SPIn-NeRF [20], lever-
aged the LPIPS metric [21] to measure perceptual similarity
rather than pixel-level accuracy for the inpainted region. While
subsequent research [22, 23, 24, 25] also adopted the LPIPS
metric to blend inconsistent inpainting results more seamlessly,
it remains unsatisfactory due to the loss of fine-grained details
and its vulnerability to discrepancies beyond the perceptual
level.
While image inpainting techniques are being used to
reconstruct occluded regions seamlessly, it is notable that
inpainting does not aim to find the ‘correct’ solution but instead
provides one of many plausible outcomes. The larger the areas
required for inpainting, the more diverse the possible outcomes.
Consequently, expecting the inpainting output to maintain per-
arXiv:2507.08434v1  [cs.CV]  11 Jul 2025

<!-- page 2 -->
(Infusion) Renderings
(Gaussian Grouping) Renderings
(SPIn-NeRF*) Renderings
Original Scene
Inpainted images
…
Reference-guided 3D scene inpainting on GS
Our Renderings
Our Method
Previous Methods
Reference
𝑤𝑤𝑡𝑡
Reference
𝐿𝐿𝐷𝐷+ 𝐿𝐿𝑖𝑖𝑖𝑖+ 𝐿𝐿𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖+ 𝐿𝐿𝑐𝑐𝑐𝑐𝑐𝑐𝑐𝑐𝑐𝑐
Pseudo-GT
GT (RGB | Depth)
 
 
Figure 1: Our method provides high-fidelity and view-consistent 3D scene inpainting using 3DGS, even in complex scenes with severe inconsistencies across
inpainted views. Given a user-selected reference view among inpaintings, our method guides 3DGS to preserve the reference appearance while adaptively leveraging
other views based on their reliability. Arrows indicate areas where the previous method underperforms. (*SPIn-NeRF results are obtained under our re-implemented
setup with conditions closely matching ours; Refer to Section 4.1 for details.)
ceptual consistency leads to inherent instability. Even without
considering the context level, image inpainting on scenes with
geometry or texture that are complex or irregular can hardly
achieve perceptual consistency across views. Despite this, very
few studies acknowledge this issue, with most merely stating
that inpainting is performed on ‘selected’ views perspective.
Mirzaei et al. [26] propose a depth-based view transformation
approach that warps a user-provided single inpainting view to
others to mitigate this issue. This input, called the reference
image, improves view consistency by filling occluded regions
in other views with a single source.
However, the method
heavily depends on the accuracy of depth estimation derived
from the reference image. While errors may not be apparent
around the reference view, they can become significant when
the viewpoint changes.
In this paper, we propose a 3D scene inpainting method that
robustly produces realistic and consistent results across mul-
tiple views, despite inconsistent image inpainting results per
view—common in complex or wide view scenes(see Fig. 1).
To achieve this, we introduce inpainting confidence evaluation
and a reference-guided 3DGS from the single inpainted image.
The reference image, selected by the user from the inpainted
views, serves as the intended target for 3D reconstruction. By
warping a reference image across multiple viewpoints, we en-
able the seamless reconstruction of occluded regions while pre-
serving 3D consistency. Furthermore, instead of relying solely
on the estimated depth from the reference image, we incorpo-
rate inpainted views based on their similarity to the reference.
We measure content-level similarity between the reference im-
age and the inpainted views after aligning them via warping,
enabling adaptive confidence weighting to improve 3D geom-
etry. By leveraging inpainting confidence evaluation and the
reference-guided approach, our method addresses a challenge
that previous studies have struggled to overcome. Our approach
leverages a single-reference inpainted image to infer hidden ar-
eas, ensuring that inpainted regions remain realistic and coher-
ent across multiple viewpoints. This method significantly re-
duces inconsistencies, providing a robust solution for complex
3D scene editing where occluded information is unavailable.
Our contributing points are as follows:
• Introduce a reference-guided 3D scene inpainting method
that leverages confidence-based multi-view fusion and ref-
erence image warping to achieve realistic and perceptually
consistent reconstruction in complex scenes.
• Estimate the confidence of multi-view inpainting results
from the reference image to guide the construction of ac-
curate, reference-aligned geometry.
• Demonstrate superior 3D scene inpainting performance on
complex backgrounds compared to conventional inpaint-
ing methods, particularly in wide-view and irregular cases.
2. Related Works
The 3D scene inpainting method for radiance fields leverages
image inpainting techniques to synthesize a new spatial lay-
out on the 3D scene. While 3D scene inpainting may involve
modifying styles or inserting new objects, this paper specifi-
cally addresses the recovery of backgrounds exposed by object
removal. Before delving into this, it is important to note that
2

<!-- page 3 -->
𝐿𝐿𝐷𝐷
𝐿𝐿𝐶𝐶
𝐿𝐿𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖
𝐿𝐿𝑖𝑖𝑖𝑖
Initial 3DGS
Rendering
RGB image
Segmentation
Image 
Inpainting
Depth 
Estimation
RGB image
Depth map
Normal map
Confidence Evaluation
Reference image
(GT) Depth
(GT) Normal
(Pseudo GT) Image
Depth map
Input views
Masks
Mono depths
Inpaint 3DGS
Masks
Inpainted images
Mono depths
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤re𝑓𝑓→1
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤1
(GT) Depth
(GT) Image
𝐿𝐿𝐶𝐶
𝐿𝐿𝐷𝐷
Rendering
RGB이미지밝기를+20, 40% 키운버전
Rendering
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤𝑟𝑟𝑟𝑟𝑟𝑟→2
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤2
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤3
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤𝑟𝑟𝑟𝑟𝑟𝑟→3
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓1
LPIPS
proc.
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓2
LPIPS
proc.
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓3
LPIPS
proc.
…
Preprocessing
Progressive Warping
Figure 2: Overview of our proposed method. First, the method performs an Initial 3DGS to reconstruct the background 3D scene while excluding the target object.
This allows rendering the background behind the object, reducing unnecessary inpainting. Next, segmentation and inpainting are applied to the incomplete regions,
which are primarily heavily occluded areas. Among the inpainted images, the user defines one as the reference image to guide the 3D reconstruction. Based on
the inpainted results and the reference, the inpainting confidence evaluation computes the reliability of each view with respect to the reference image warped to the
same viewpoint. Finally, the Inpaint-3DGS reconstructs the 3D inpainting by guiding the inpainting regions closer to the reference. This is done by adjusting each
view’s weight based on their confidence and warping the reference view to serve as a pseudo-ground truth.
𝐿𝐿𝐷𝐷
𝐿𝐿𝐶𝐶
𝐿𝐿𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖
𝐿𝐿𝑖𝑖𝑖𝑖
Initial 3DGS
Rendering
RGB image
Segmentation
Image 
Inpainting
Depth 
Estimation
RGB image
Depth map
Normal map
Confidence Evaluation
Reference image
(GT) Depth
(GT) Normal
(Pseudo GT) Image
Depth map
Input views
Masks
Mono depths
Inpaint 3DGS
Masks
Inpainted images
Mono depths
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤re𝑓𝑓→1
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤1
(GT) Depth
(GT) Image
𝐿𝐿𝐶𝐶
𝐿𝐿𝐷𝐷
Rendering
RGB이미지밝기를+20, 40% 키운버전
Rendering
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤𝑟𝑟𝑟𝑟𝑟𝑟→2
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤2
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤3
𝑣𝑣𝑖𝑖𝑖𝑖𝑤𝑤𝑟𝑟𝑟𝑟𝑟𝑟→3
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓1
LPIPS
proc.
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓2
LPIPS
proc.
𝑐𝑐𝑐𝑐𝑐𝑐𝑓𝑓3
LPIPS
proc.
…
Preprocessing
Progressive Warping
Figure 2: Overview of our proposed method. First, the method performs an Initial 3DGS to reconstruct the background 3D scene while excluding the target object.
This allows rendering the background behind the object, reducing unnecessary inpainting. Next, segmentation and inpainting are applied to the incomplete regions,
which are primarily heavily occluded areas. Among the inpainted images, the user defines one as the reference image to guide the 3D reconstruction. Based on
the inpainted results and the reference, the inpainting confidence evaluation computes the reliability of each view with respect to the reference image warped to the
same viewpoint. Finally, the Inpaint-3DGS reconstructs the 3D inpainting by guiding the inpainting regions closer to the reference. This is done by adjusting each
view’s weight based on their confidence and warping the reference view to serve as a pseudo-ground truth.
Input scene
LaMa
SDXL (view A)
SDXL (view B)
Figure 3: Limitation in image inpainting. For complex scenes, image inpainting
often fails to fill in the background seamlessly or produces semantically differ-
ent outputs across views. The results shown here are obtained using LaMa [17]
and SDXL [19], two widely used inpainting models.
standard radiance field methods assume a static scene and con-
sistent input across views, enabling stable 3D reconstruction
based on view poses and image data. However, the image in-
painting technique is performed individually on images with-
out considering 3D information, resulting in differences across
views. Studies have proposed diverse methods to address the
issue of inconsistent inpainting results. We categorize these
approaches into the following four parts: view selection, 3D
segmentation, 3D optimization, and geometric transformation.
View Selection. View selection refers to selecting appropri-
ate views for reconstructing the inpainted regions. Filtering out
inpainting results that have failed or significantly differ from
other views can help reduce inconsistencies. Though recent im-
age inpainting techniques [17, 19] show high-quality and stable
(a) Inpainted image w/ Mask
(b) LPIPS-optimized result
(c) Inpainted image w/ Mask
(d) LPIPS-optimized result
Figure 4: Limitation in LPIPS metric. LPIPS performs well on regular pat-
terns when optimizing multi-view inpainting results, as it captures shared visual
structures across views beyond pixel-wise differences. However, it performs
poorly on irregular ones, where achieving consistent visual structures across
views is challenging. (a) and (c) show samples of inpainted input views with
regular and irregular patterns, respectively, while (b) and (d) present the corre-
sponding results.
performance across various scenes, they may fail in complex
scenes or under unfamiliar view directions (see Fig. 3). More-
over, inconsistencies across multi-view can range from minor
details to content level. Despite this, most studies do not handle
how to select appropriate training samples in the presence of
inconsistent inpaintings. Some merely mention that the exper-
iments were conducted on ‘user-chosen’ views without further
explanation [27]. Weder et al. [28] attempts to adjust contribu-
3
Figure 3: Limitation in image inpainting. For complex scenes, image inpainting
often fails to fill in the background seamlessly or produces semantically differ-
ent outputs across views. The results shown here are obtained using LaMa [17]
and SDXL [19], two widely used inpainting models.
standard radiance field methods assume a static scene and con-
sistent input across views, enabling stable 3D reconstruction
based on view poses and image data. However, the image in-
painting technique is performed individually on images with-
out considering 3D information, resulting in differences across
views. Studies have proposed diverse methods to address the
issue of inconsistent inpainting results. We categorize these
approaches into the following four parts: view selection, 3D
segmentation, 3D optimization, and geometric transformation.
View Selection. View selection refers to selecting appropri-
ate views for reconstructing the inpainted regions. Filtering out
inpainting results that have failed or significantly differ from
other views can help reduce inconsistencies. Though recent im-
age inpainting techniques [17, 19] show high-quality and stable
(a) Inpainted image w/ Mask
(b) LPIPS-optimized result
(c) Inpainted image w/ Mask
(d) LPIPS-optimized result
Figure 4: Limitation in LPIPS metric. LPIPS performs well on regular pat-
terns when optimizing multi-view inpainting results, as it captures shared visual
structures across views beyond pixel-wise differences. However, it performs
poorly on irregular ones, where achieving consistent visual structures across
views is challenging. (a) and (c) show samples of inpainted input views with
regular and irregular patterns, respectively, while (b) and (d) present the corre-
sponding results.
performance across various scenes, they may fail in complex
scenes or under unfamiliar view directions (see Fig. 3). More-
over, inconsistencies across multi-view can range from minor
details to content level. Despite this, most studies do not handle
how to select appropriate training samples in the presence of
inconsistent inpaintings. Some merely mention that the exper-
iments were conducted on ‘user-chosen’ views without further
explanation [27]. Weder et al. [28] attempts to adjust contribu-
3

<!-- page 4 -->
tion per view by estimating the inpainting confidence as a loss
attenuation term, similar to aleatoric uncertainty. However, this
confidence fails to select reliable views when the majority of
inpainted views lack mutual consistency, making it particularly
vulnerable to diverse inpainting completions caused by large
masked regions.
3D segmentation. 3D segmentation involves separating the
target object from the background in the 3D scene rather than
segmenting it directly in 2D images for inpainting. While early
studies in object removal in radiance fields applied image in-
painting to the entire regions containing target objects per view,
Huang et al. [23] pointed out that applying inpainting even to
parts occluded in one view but visible in another is unneces-
sary. To address this, they propose segmenting the target ob-
ject during the initial 3DGS training, which enables extract-
ing the reconstructed background behind the object and thus
reduces unnecessary inpainting. A related idea is seen in Gaus-
sian Grouping [25] and Infusion [29]. In the case of Gaussian
Grouping, it jointly reconstructs and segments components of
the scene and then supports object removal for each instance.
3D segmentation within the radiance field helps reduce unnec-
essary inpainting, effectively narrowing down the regions that
3D scene inpainting needs to address. However, further solu-
tions are needed to handle inconsistent inpainting in heavily
occluded regions.
3D optimization. Regardless of the proposed preprocessing
steps, 3D scene inpainting still requires an optimization process
to integrate inconsistent inpainting results across views into a
coherent and unified space. Differences in inpainting across
views can confuse the geometry estimation process in the radi-
ance field, leading to artifacts such as floating or hallucinations.
To mitigate this, NeRF-In [27] proposes utilizing depth inpaint-
ing to assist geometry construction in radiance fields. They op-
timize the depth loss to enforce the geometry with the inpainted
depth. Since implementing a 3D scene relying solely on im-
age inpainting was unstable, depth inpainting has been widely
adopted in subsequent related studies to stabilize the 3D ge-
ometry to some extent [22, 28, 20, 23]. For depth inpainting,
various estimation methods have been applied and can be sub-
stituted. One commonly used model is LaMa [17], which was
used for RGB image inpainting. Mirazaei et al. [20] found that
the same LaMa used for RGB inpainting also provides suffi-
ciently high-quality results for depths. These are also demon-
strated in experiments by Fischer et al. [30], which compare
different deep neural networks for depth inpainting. However,
while LaMa approximates depth in unknown regions based on
the surrounding context, the resulting geometry is often inaccu-
rate. Pointing out these limitations, Liu et al. [29] propose us-
ing an image-conditioned depth completion model that directly
restores the depth map from the RGB image. This method esti-
mates depth in a way that is more semantically consistent with
the inpainted content by leveraging surrounding depth cues and
RGB context. However, while depth inpainting assists 3D re-
construction under inconsistent image inpainting, the problem
of reconstructing fine geometric details and coherently integrat-
ing their appearance still remains.
Not to mention the semantically different inpaintings, even
slight differences across views, such as minor distortions or
shifts, can be fatal for optimizing radiance fields-based methods
when comparing pixel-to-pixel differences. Therefore, Mirzaei
et al. [20] propose using perceptual similarity for the inpainted
region. The LPIPS metric [21] measures the content similarity
between images in a way that aligns with human visual percep-
tion, allowing for distortions, noise, lighting changes, shifts,
and color variations within a perceptible range.
Comparing
the similarities between contexts can smoothly gather the sub-
tle differences between inpaintings as a coherent form. Due
to its ability to integrate inconsistent inpainting inputs into a
consistent perceptual context, the use of LPIPS loss has been
widely adopted in subsequent studies [22, 24, 23, 31, 25]. The
LPIPS metric is effective when the inpainted train set is in per-
ceptual similar relationships, as referred in Fig. 4(b). However,
in practice, the inpainting results are not deterministic due to the
nature of the generative models. Achieving consistent inpaint-
ing outcomes across multiple views is challenging, as image
inpainting can yield varying outputs not only in details but also
in semantic content. For instance, the SDXL results in Fig. 3
show that, despite visually convincing complex scenes, the out-
puts can differ semantically even between closely related view-
points. As shown in Fig. 4(d), LPIPS struggles with complex
textures where consistent outputs are difficult to achieve.
Geometric transformation. To avoid the inherent limita-
tions of inpainting, an alternative approach has been proposed
that estimates depth from one or a few inpainted views and re-
stores the scene through geometric transformation. One repre-
sentative approach is by Mirzaei et al. [26], who propose us-
ing a single inpainted image as a reference and transforming
it to other views based on its estimated depth. This approach
reduces the amount of inpainting required for other views by
limiting it to view-specific occluded areas, thereby minimizing
inconsistencies across views. However, the depth of the ref-
erence image is estimated using monocular depth estimation,
which may be less accurate. These errors can become more
pronounced as the viewpoint moves away from the reference
view. Infusion [29] also proposes utilizing user-selected refer-
ence inpainting to improve fidelity. Instead of relying on cross-
validation across multiple inpaintings to refine the geometry,
their approach establishes a plausible initial geometry by un-
projecting the inpainted depth into a 3D Gaussian point cloud.
This serves as a robust initialization for subsequent GS fine-
tuning. However, this approach also relies on estimated depth
from one or a few reference views, and inaccuracies in these
estimates can lead to distorted forms from other viewpoints. In
other words, successful geometric transformation of the refer-
ence image into a 3D scene or other views requires validating
the estimated depth of the selected views.
3. Methods
Our proposed method consists of three key processes: initial
3DGS, inpainting confidence evaluation, and inpaint-3DGS.
The pipeline of our proposed approach is illustrated in Fig. 2.
The first key process of our method, initial 3DGS (section 3.1),
4

<!-- page 5 -->
reconstructs the scene without the target object to reveal par-
tially occluded backgrounds.
We then apply state-of-the-art
(SOTA) image inpainting and monocular depth estimation mod-
els to the rendered background images. Out of the inpainted
results, the user selects a reference image which will provide a
consistent visual guideline for how the target object should ap-
pear. The remaining inpainted views are compared with this ref-
erence to compute their content-level similarity. This similarity
serves as an indicator of how reliably each inpainting can recon-
struct the scene in accordance with the reference, a process we
refer to as inpainting confidence evaluation(section 3.2). Based
on the computed confidence scores, inpaint-3DGS optimizes
the geometry of the inpainted region while reconstructing its
appearance using warping techniques (section 3.3). Ultimately,
our method integrates inconsistent inpaintings into a coherent
3D reconstruction that faithfully preserves high-frequency vi-
sual details from the reference image.
3.1. Initial 3DGS: Removal 3DGS
Before attempting to inpaint the occluded region behind the
target object, initial 3DGS is conducted on the original images
with the segmented masks to reconstruct the background 3D
scene.
Initial 3DGS builds upon vanilla 3DGS [4], adopting its over-
all pipeline while introducing modifications specific to our task.
The color C of a pixel is computed by blending a depth-ordered
set N of points overlapping the pixel, considering each point’s
color ci and its’ opacity αi, and transmittance from pixel to that
point.
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −α j)
(1)
From this, we obtain the initial 3DGS by excluding the masked
target region during reconstruction, thereby producing a 3D
background scene without the target object.
For the color loss of the original region, the rendered image
ˆI is compared with input image I at corresponding view using
L1 loss with a D-SSIM term, excluding the masked region M:
LoC = (1 −λssim) 1
|Mc|
X
p∈Mc
I(p) −ˆI(p)

+λssim(1 −SSIM(I|Mc, ˆI|Mc))
(2)
Here, I(p) denotes a pixel value at position p; Mc denotes the
complement of the masked region; |Mc| is the number of pixels
in that set; I|Mc is the restrictions of I to the unmasked region
Mc; and λssim ∈[0, 1] is a weighting coefficient. Completely
occluded regions across all views appear as holes, whereas par-
tially occluded areas visible in other views can be recovered.
However, even if a region is partially visible, reconstruct-
ing it may be difficult when the visible views are insufficient.
In general, neural rendering struggles with limited view di-
versity, biased viewpoints, or visually simple indoor scenes.
Such conditions often suffer from inaccurate geometry due to
limited constraints or view diversity. Accordingly, to provide
regularized optimization under sparse constraints, many stud-
ies have proposed using depth priors—such as sparse points
from Structure-from-Motion (SfM) [32], monocular depth es-
timation [33, 34, 35], or depth completion [36]—to guide the
optimization better and boost reconstruction accuracy. To this
end, we utilize monocular depth estimates to guide geometry
reconstruction during the initial 3DGS stage, enabling better
generalization in diverse situations. Monocular depth provides
context-aware dense estimates from a single image, whereas
SfM yields only sparse depths at matched feature points, and
depth completion struggles in large missing regions. However,
considering the inherent scale ambiguity and object-level noise
in monocular depth estimation, we align the estimated depth
maps to the COLMAP [37] sparse point cloud, following DN-
Splatter [34]. By applying the closed-form linear regression
solution, optimal scale a and shift b parameters for each image
are found.
ˆa, ˆb = arg min
a,b
X
p∈Pspc
(aDmono(p) + b) −Dspc(p)
2
(3)
Here, Pspc is the set of pixels corresponding to the sparse point
cloud, with Dmono and Dspc denoting the monocular estimated
depth and the sparse point cloud depth, respectively. During
the optimization, per-pixel z-depth D was estimated similar to
colors, but using ith Gaussian’s z-depth di in view space instead
of its’ color.
D =
X
i∈N
diαi
i−1
Y
j=1
(1 −α j)
(4)
With the rendered depth ˆD of the specific view, corresponding
estimated depth Dmono is compared. The depth loss of the orig-
inal region is formulated as follows:
LoD =
1
|Mc|
X
p∈Mc
Dmono(p) −ˆD(p)

(5)
Here, the estimated depth serves as auxiliary guidance, improv-
ing structural accuracy.
Initial 3DGS reduces the unnecessary inpainting in regions
that are occluded but visible from other points of view. Ren-
dered background images are used to generate preprocessing
data for 3D inpainted scene reconstruction. Using SOTA mod-
els such as Segment Anything Model 2 [38], Stable Diffusion
XL [19], and Depth Anything v2 [39], we sequentially gener-
ate a reduced mask by segmenting unseen regions, then perform
image inpainting, and estimate depth. The estimated depth, cur-
rently obtained from monocular estimation, is then aligned to
the sparse point cloud from SfM.
3.2. Inpainting Confidence Evaluation
Considering that inpainting results can vary significantly
from low-level details to high-level semantics, we evaluate the
reliability of each view’s inpainting based on a reference im-
age.
Among the inpainted views generated in the previous
stage, where background images were rendered from the ini-
tial 3DGS and then inpainted, a reference image is selected by
the user. This reference reflects the target that the user intends
to build. Unlike previous methods [26, 29] that rely solely on
5

<!-- page 6 -->
the accurate depth estimation from a few reference views, our
approach utilized other inpainted views—not indiscriminately,
but by weighting their influence according to their similarity to
the reference. Specifically, we estimate the reliability of each
inpainted view by warping the reference image and measuring
content-level similarity, enabling adaptive weighting in the re-
construction process. This process can be seen in Fig. 5.
3.2.1. Warping
Image warping is the process of manipulating an image by
distorting its shape. It is also used to geometrically transform
an image based on correspondences such as depth, simulat-
ing how the image would appear from a different viewpoint.
Through this process, different inpainting results are geomet-
rically aligned to a common view, thereby facilitating a struc-
tural similarity comparison. However, since some regions may
be occluded or outside the field of view in certain views, we
assume that most regions requiring inpainting are visible in the
reference view. The reference image is warped to the inpainted
region of the target view using its depth. Based on the depth
map Dj from view j, the corresponding point p in view i can be
computed as follows:
pj→i ≈KRiR−1
j Dj(pj)K−1p j
(6)
where K denotes the camera intrinsic parameters, and Ri, R j
represent the extrinsic matrices of views i and j, respectively.
To handle occlusions that can lead to many-to-one matching,
the pixel closest to the target camera is selected as the match-
ing point using Z-buffering. Unmatched pixels after warping
are filled via bilinear interpolation using nearby valid samples,
enabling smooth propagation of image content.
3.2.2. Confidence evaluation
Joint
Bilateral Filter
Joint
Bilateral Filter
Depth-based 
Warping
Compute
confidence
(1-LPIPS)
𝑐𝑐𝑜𝑜𝑜𝑜𝑓𝑓𝑡𝑡𝑡𝑡𝑡𝑡
𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝ℎtar
𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝ℎref
𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝ℎref→tar
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤𝑡𝑡𝑡𝑡𝑡𝑡
𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝ℎtar
𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝ℎ𝑟𝑟𝑟𝑟𝑟𝑟→𝑡𝑡𝑡𝑡𝑡𝑡
Inpainted image
Mono depth
Inpainted image
Mono depth
𝑣𝑣𝑣𝑣𝑣𝑣𝑤𝑤𝑟𝑟𝑒𝑒𝑒𝑒
Figure 5: Confidence evaluation of inpainted region. Each region’s confidence
is assessed by warping a reference view to each target view using their esti-
mated depth maps and comparing their content-level similarity. To reduce the
influence of fine texture differences, a strong bilateral filter is applied before-
hand to suppress details on smooth surfaces.
Once the inpainted images are aligned to the same view, their
similarity can be assessed. As multi-view inpainted images are
used to construct geometry that supports stable image warp-
ing, our focus is on geometric consistency with the reference
image rather than subtle pixel-level differences. For this rea-
son, we use the perceptual similarity metric LPIPS [21], which
measures structural similarity between images. It provides a
way to quantitatively assess visual resemblance in a manner
closer to human perception and shows robustness to distortions
or color differences, unlike traditional metrics (e.g., L2, PSNR,
SSIM). We extract a patch region P from an enlarged bounding
region that fully contains the inpainted area in the two views be-
ing compared. However, perceptual similarity also responds to
variations in surface texture, which is not desirable for our pur-
pose. To suppress fine textures on smooth surfaces, we apply
strong bilateral filtering [40] to the inpainted images, guided by
the depth and normal maps derived from estimated depth. We
denote the bilateral-filtered image as IBF. With detailed texture
flattened, the perceptual similarity is evaluated between the in-
painting of the target view patch I j|P and the warped reference
patch Ire f→j|P, both restricted to the patch region.
confj = 1 −LPIPS (IBF
re f→j|P, IBF
j |P)
(7)
The estimated confidence reflects the quality of both RGB and
depth inpaintings of the view. Since RGB and depth inpaintings
are evaluated as a pair, imprecise depth estimation or alignment
that causes substantial distortion during warping can lead to low
confidence despite the similarity in RGB inpainting. To mini-
mize such cases, each view’s confidence is periodically updated
during inpaint-3DGS by refining the alignment between the
estimated depth and the geometry progressively reconstructed
during training. Based on the confidence, each view is weighted
and incorporated into training inpaint-3DGS. Incorporating in-
paintings from highly relevant views helps mitigate the limita-
tions of single-view dependency by providing complementary
information.
3.3. Inpaint-3DGS
As the final step, inpaint-3DGS is performed to reconstruct
previously hidden regions based on the inpainted data.
For the region outside the mask, the optimization largely fol-
lows the standard strategy of Initial 3DGS, minimizing the pho-
tometric loss based on pixel-wise color differences (Eq. (2)) and
applying depth regularization on estimated depth (Eq. (5)) to
further improve the overall geometric quality. However, slight
modifications are made to the loss formulation to handle the
inpainting regions.
To reconstruct the inpainted scene, we incorporate losses de-
signed to align the inpainted region with a user-selected ref-
erence image, promoting cross-view consistency. We aim to
build perceptually coherent geometry across multiple reliable
inpainted views and then transfer the appearance of the refer-
ence image via warping. Thus, we introduce losses focusing on
two complementary objectives: constructing the geometry and
aligning the appearance with a reference image. Please note
that the loss is evaluated independently for each view, randomly
sampled at each iteration during optimization. For notational
simplicity, we omit explicit indices for the view in the formu-
lation. The loss on the inpainted region is modulated by the
inpainting weight winp = σ(α(conf −β)), where σ(·) is the sig-
moid function, and α, β ∈R control the scale and offset of the
input, respectively.
6

<!-- page 7 -->
The depth loss is formulated as follows, taking into account
both the inpainted region M and the original region Mc:
LD = 1
|D|
X
p∈Mc
Dmono(p) −ˆD(p)

+ winp
|D|
X
p∈M
Dmono(p) −ˆD(p)

(8)
The depth loss helps suppress floating artifacts caused by
inconsistent inpainting input, promoting stable and consistent
geometry reconstruction. However, the estimated depth is a
rough guide for placing Gaussians near the ground-truth surface
rather than indicating their precise positions. Commonly, neu-
ral rendering methods suffer from artifacts and ambiguities due
to the lack of 3D cues and surface constraints. To mitigate this,
some approaches [34] jointly supervise both depth and normal
to produce visually and geometrically plausible 3D reconstruc-
tions suitable for mesh conversion. Normal regularization en-
courages Gaussians to better conform to the underlying scene
geometry. Accordingly, we employ a normal loss to enhance
geometric consistency, computed on a random patch region P
whose center is randomly sampled from the inpainting region.
LiN = winp(1 −CosS im(Nmono|P, ˆN|P))
(9)
It is computed via cosine similarity between Nmono|P and ˆN|P,
the normals derived from estimated and rendered depth, respec-
tively.
For appearance supervision, we overlay the warped reference
image onto the current view and use it as a pseudo-ground truth
to compute LPIPS loss and color loss.
However, when the constructed geometry is yet inaccurate,
leveraging warped outputs as pseudo-ground truth can further
amplify distortion. To prevent this issue, we selectively warp
regions based on the gap between the rendered depths, inspired
by the approach of GeCoNeRF [41]. GeCoNeRF warps input
images to unseen views to perform regularization from sparse
views. In this process, regions where the reprojected depth de-
viates from the corresponding view’s depth beyond a certain
threshold are treated as occluded and excluded from the com-
parison. Following this idea, we exclude pixels from warping
when the gap between the two aligned depths exceeds a thresh-
old, treating them as geometrically inconsistent regions. As the
training progresses and the geometry improves, the warped area
progressively expands (see Fig. 2). Formally, we define the set
of geometrically consistent pixels P′ as:
P′ =
n
p ∈P ∩M | | ˆDref(p) −ˆDtar→ref(p)| ≤τ
o
where ˆDtar→ref is the rendered depth from the target view re-
projected to the reference view, and τ is the threshold for con-
sistency. Using the set P′, the warped reference image Iref→tar
is blended into the target view Itar. The resulting image Iwarp
is defined as follows: Within P′, Iwarp is obtained by solving
the Poisson equation with the Laplacian of Iref→tar as the guid-
ance field, with Dirichlet boundary conditions imposed such
that Iwarp = Itar on the boundary of P′. Outside P′, Iwarp is
set to Itar. This formulation ensures that the inserted content
blends smoothly into the surrounding region of the target im-
age, preserving second-order structural consistency and elimi-
nating visible seams [42].
𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝐷𝐷𝑡𝑡𝑡𝑡𝑡𝑡
𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝑟𝐷𝐷𝑟𝑟𝑟𝑟𝑟𝑟
V𝑖𝑖𝑖𝑖𝑤𝑤𝑡𝑡𝑡𝑡𝑡𝑡
V𝑖𝑖𝑖𝑖𝑤𝑤𝑟𝑟𝑟𝑟𝑟𝑟
Figure 6: Geometry-Consistent Warping. When warping a reference view to a
target view, only regions with rendered depth differences below a threshold are
deemed geometrically consistent and transferred. During optimization, warping
is progressively applied, leaving out inconsistent regions.
Based on the warped image, LPIPS and color loss are utilized
to optimize the image’s overall appearance and fine details.
Lilpips = winpLPIPS(Iwarp|P, ˆI|P)
(10)
LC = 1 −λssim
|Mc ∪P′|
X
p∈Mc∪P′
Iwarp(p) −ˆI(p)

+ λssim(1 −SSIM(Iwarp|Mc∪P′, ˆI|Mc∪P′))
(11)
It should be noted that LPIPS loss, Lilpips, is computed over
the entire inpainting patch region P to help build a rough ini-
tial shape from the non-uniform inpainting. In contrast, the
color loss is computed only over the warped regions P′ together
with the original background region Mc to refine the fine details
of the warped reference information that are deemed geometri-
cally consistent.
Overall, the total loss for optimization is defined as follows:
Ltotal = LC + λDLD + λiNLiN + λiLPIPSLiLPIPS
(12)
Such an approach refines the intended geometry of the in-
painted regions using depth and normal estimates weighted by
reliability, while simultaneously guiding the appearance toward
the reference through comparisons with progressively warped
images, resulting in a realistic and perceptually consistent 3D
scene.
4. Experiments
In this section, we evaluate our method on forward-facing
and wide-view datasets. Our method is compared to existing
3D scene inpainting methods, followed by an ablation test.
7

<!-- page 8 -->
Table 1: Comparison of baseline method components related to inpainted region construction, and our experimental settings for fair comparison. Specifically,
SPIn-NeRF (reimpl.) was adapted to 3DGS and evaluated using the shared mask and inpainted images generated via 3D segmentation in 3DGS.
Category
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
Inpainting-Related
Method Details
Rendering Base
3DGS
(Original: NeRF)
3DGS
3DGS
3DGS
3D segmentation
Yes
(Original: None)
Yes
Yes
Yes
View usage
All
(weight: uniform)
All
(weight: uniform)
Reference views
only
All + Reference
(weight: confidence)
Reference
Integration
N/A
N/A
Estimated
depth-based
reprojection
Optimized
depth-based
warping
Optimization
- LPIPS loss
- Depth loss
- LPIPS loss
- Color loss
- Color loss
- LPIPS loss
- Depth loss
- Normal loss
Experimental
Setting
Mask
SAM2 on our initial 3DGS render
Inpainted image
SDXL inpainting on our initial 3DGS render
Estimated depth
Depth anything v2
N/A
Diffusion-based
completion
(Infusion)
Depth anything v2
4.1. Implementation details
Our code for initial 3DGS and inpaint-3DGS was built upon
the vanilla 3DGS. For data preparation, three tasks are required
in our method: 2D segmentation, image inpainting, and depth
estimation. For 2D segmentation, we utilized Segment Any-
thing Model 2 (SAM2) [38] to generate segmentation masks of
the target object. SAM2 provides video segmentation capabil-
ity, which aligns well with multi-view image sets typically used
in neural rendering, as they exhibit overlapping regions resem-
bling sequential frames. With a point prompt annotated on a
single image, SAM2 can propagate the mask throughout the se-
quence. However, due to occasional loss of detail in complex
geometries, we expand the masks to ensure better coverage. For
image inpainting, we used Stable Diffusion XL (SDXL) [19], a
latent diffusion model for text-to-image synthesis that shows
robust performance due to its large U-Net backbone architec-
ture compared to Stable Diffusion [43]. Lastly, we used Depth
Anything v2 (Metric Depth version) [39] for monocular depth
estimation. In our method, all monocular depth estimates are
first aligned to either COLMAP’s sparse points or the rendered
depth from 3DGS before use. However, the models can be
substituted with other alternatives. Our experiment was im-
plemented on an NVIDIA A100 discrete GPU with 40GB of
memory.
4.2. Experiment Setups
Datasets For experimental evaluation, we used a total of 10
sample scenes from three datasets: five from SPIn-NeRF [20],
two from MipNeRF360 [2], and three from our custom dataset.
For qualitative evaluation, we utilized the SPIn-NeRF dataset,
which provides real-world images containing the removal target
along with clean background images as ground truth. Among
the scenes, we selected five with complex structures or textures
that relatively showed larger inpainting discrepancies between
views. Each scene comprises 60 training views containing the
target object and 40 test views depicting the background after
object removal. However, to demonstrate the effectiveness of
our reference-guided 3D scene inpainting method, we use one
image from the test set as the reference. Ultimately, training
was performed on 61 views, with the remaining 39 views were
used for evaluation. However, the SPIn-NeRF dataset is limited
to forward-facing scenes and thus insufficient to demonstrate
robust performance across diverse viewpoints. Therefore, we
additionally employed two 360◦view real-world scenes from
MipNeRF360 to visually showcase our performance across di-
verse viewpoints. Each scene contains 292 and 185 images,
respectively. We split the views using 3/4 for training and the
rest for testing.
Furthermore, to qualitatively assess the robustness of our
method across a wide range of viewpoints and in complex
scenes, we created a custom synthetic dataset of three scenes
using Blender [44] and assets from BlenderKit [45].
These
scenes were designed to contain complex structures or textures,
making it challenging to achieve consistent inpainting across
views. Each scene contains a number of training and test views
as follows: Scene 1 (75 / 33), Scene 2 (32 / 53), and Scene 3
(107 / 53).
Baseline Among related studies, we identified and selected
the following methods as baselines based on their relevance to
8

<!-- page 9 -->
our approach. Firstly, SPIn-NeRF [20] as it is one of the most
prominent approach for object removal in radiance fields. It op-
timizes the inpainted region using perceptual similarity, empha-
sizing the overall structure of the image rather than fine-grained
inconsistencies, which is also adopted in our work.
However, SPIn-NeRF is based on NeRF, and a notable per-
formance gap exists between NeRF and 3DGS. Therefore, to
compare under similar conditions, we re-implemented SPIn-
NeRF on 3DGS by closely following the procedure described
in the original paper.
When comparing the LPIPS scores on the SPIn-NeRF sam-
ple datasets using the LaMa inpainting technique [17], our
3DGS-based reimplementation achieved 0.394, outperforming
the original implementation’s 0.487. This lower perceptual er-
ror supports the validity of our comparison.
Additionally, Gaussian Grouping [25], which minimizes the
inpainting region by performing 3D segmentation in advance,
was included as a baseline for comparison.
The integration
method is similar to SPIn-NeRF but does not utilize depth in-
painting. Finally, we included Infusion [29] in our compari-
son, which achieves multi-view consistency by reprojecting es-
timated depth from reference images onto the segmented 3D
Gaussian scene. The Table 1 summarizes key components re-
lated to inpainted region construction and experimental setup.
Metrics To evaluate the quality of 3D scene inpainting, we
referred to the evaluation criteria used in prior studies on ob-
ject removal in radiance fields [28, 22, 20, 26, 29]. We mainly
employed LPIPS [21], FID [46], and SSIM [47] to assess per-
ceptual similarity, distributional similarity, and structural con-
sistency by comparing the rendered image with the ground truth
image in the test set.
4.3. 3D Scene Inpainting Evaluation
We conduct experiments on SPIn-NeRF, MipNeRF360, and
our custom dataset, comprising a total of 10 scenes. However,
it should be noted that, in an effort to ensure equal conditions,
the inpainted images generated after our initial 3DGS were pro-
vided to all baseline methods. In this experiment, SPIn-NeRF
can also be regarded as undergoing a 3D segmentation process.
Fig. 7 shows the qualitative comparisons with baseline meth-
ods. The images are rendered from one of the test views. It
can be seen that our method plausibly and finely reconstructs
the background appearance, producing inpainted regions that
blend naturally with the original scene.
On the other hand,
while SPIn-NeRF generally captures the overall semantic con-
tent well, it fails to produce fine details and performs poorly on
challenging scenes with severe inpainting inconsistencies, such
as the bench scene (Fig. 7(a)). In the case of Gaussian Group-
ing, floating artifacts appear depending on the level of inconsis-
tency in the input, as it does not utilize depth inpainting. Lastly,
for Infusion, the view selected for depth completion was set to
be the same as the reference view used in our method. Although
Infusion produced visually plausible results for views close to
the selected, errors in depth estimation became more apparent
as the viewpoint moved further away. Such behavior is more
clearly illustrated in Fig. 8, which compares results on three
different novel views across several wide-range view datasets.
Unlike baselines that lose visual details or cause geometric dis-
tortions, our method preserve realistic and consistent content
across multiple viewpoints.
Table 2: Qualitative results on the SPIn-NeRF dataset. Average values were
calculated from five scenes. Measurements were computed within the bounding
box of the inpainted region.
Method
SSIM ↑
LPIPS ↓
FID ↓
SPIn-NeRF (reimpl.)
0.3696
0.3521
174.62
Gaussian Grouping
0.2795
0.4409
218.84
Infusion
0.4313
0.3902
154.02
Ours
0.4740
0.2431
70.73
Table 3: Qualitative results on our custom dataset. Average values were cal-
culated from three scenes. Measurements were computed within the bounding
box of the inpainted region.
Method
SSIM ↑
LPIPS ↓
FID ↓
SPIn-NeRF (reimpl.)
0.7659
0.2776
153.23
Gaussian Grouping
0.7352
0.2994
165.31
Infusion
0.7601
0.3544
194.50
Ours
0.7913
0.2267
104.00
We also evaluate the quantitative performance of our method.
To assess the quaility of 3D scene inpainting, the evaluation was
conducted within the bounding box of the inpainting region.
Table 2 and Table 3 present quantitative comparisons against
baseline methods. These results show that our proposed method
performs best in terms of SSIM, LPIPS, and FID.
Moreover, Fig. 9 shows that our method produces results
that remain perceptually faithful to the single reference across
a wide range of viewpoints. Although the overall appearance
is slightly darker due to shadows left by the removed object, a
consistent appearance with the reference image is maintained.
4.4. Ablation studies
To assess the contribution of each component in our pro-
posed methods, we conducted ablation experiments focusing on
two key aspects. One compares the performance of inpainting’s
confidence-based weighting, and the other evaluates the effect
of different loss term combinations used during optimization.
Table 4 shows the results of the ablation study on confidence-
based weighting. In the table, Uniform-Weight refers to train-
ing inpaint-3DGS with equal weights to each view, regardless
of inpainting confidence, and Confidence-Threshold denotes
the setting in which inpainted regions with confidence values
below a certain threshold are skipped during training. It shows
that adjusting each view’s weight based on their inpainting con-
fidence shows moderate improvement.
While its impact is limited when the inpainted images are
largely geometrically consistent, it offers significant improve-
ments in the presence of severe inconsistencies.
Following, Table 5 reports the impact of varying loss term
combinations in our ablation study. We ablate the losses re-
lated to the inpainted regions from the full loss formulation
9

<!-- page 10 -->
(a)
(b)
(c)
(d)
(e)
Original scene & Mask
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
Figure 7: Visual comparison of test set renderings. This figure presents sample renderings from the novel test view: (a)–(c) are from SPIn-NeRF, (d)–(e) from
MipNeRF360, and (f)–(h) from our custom dataset. Each scene is shown in two rows—the top row presents the full image, while the bottom row shows a cropped
region focusing on the inpainted area. Arrows indicate areas where the baseline underperforms. Our method not only preserves fine details from the inpainting input
but also reconstructs perceptually consistent scenes across views, guided by a single reference image. For clearer comparison, some images have been adjusted in
brightness and contrast.
10
Figure 7: Visual comparison of test set renderings. This figure presents sample renderings from the novel test view: (a)–(c) are from SPIn-NeRF, (d)–(e) from
MipNeRF360, and (f)–(h) from our custom dataset. Each scene is shown in two rows—the top row presents the full image, while the bottom row shows a cropped
region focusing on the inpainted area. Arrows indicate areas where the baseline underperforms. Our method not only preserves fine details from the inpainting input
but also reconstructs perceptually consistent scenes across views, guided by a single reference image. For clearer comparison, some images have been adjusted in
brightness and contrast.
10

<!-- page 11 -->
(f)
(g)
(h)
Original scene & Mask
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
Figure 7: Visual comparison of test set renderings. This figure presents sample renderings from the novel test view: (a)–(c) are from SPIn-NeRF, (d)–(e) from
MipNeRF360, and (f)–(h) from our custom dataset. Each scene is shown in two rows—the top row presents the full image, while the bottom row shows a cropped
region focusing on the inpainted area. Arrows indicate areas where the baseline underperforms. Our method not only preserves fine details from the inpainting input
but also reconstructs perceptually consistent scenes across views, guided by a single reference image. For clearer comparison, some images have been adjusted in
brightness and contrast.
Table 4: An ablation study on the effect of confidence-guided view weight-
ing was conducted on our custom dataset. Uniform-Weight uses equal weights
for all views, Confidence-Threshold excludes low-confidence views based on a
threshold, and Confidence-Weighted adaptively weights views by confidence.
Method
SSIM↑
LPIPS↓
FID↓
Uniform-Weight
0.7763
0.2312
107.93
Confidence-Threshold
0.7889
0.2278
104.28
Confidence-Weighted (ours)
0.7913
0.2267
104.00
(Eq. (12)). Specifically, we sequentially exclude the normal,
perceptual, and color loss on the warped reference image. The
depth loss, which is commonly used in prior work, is retained
as its effectiveness has already been well established. Here, w/o
LiC denotes the removal of the inpainting color loss component
from LC, specifically by excluding the warped reference image
from the calculation. Although LC is not strictly decomposable,
we use it for clarity in the ablation setting.
Although the final proposed method seems to show slightly
Table 5: Ablation study of our proposed loss formulation on the custom dataset.
We evaluate the contribution of each loss term by removing them individually.
w/o LiC refers to excluding the warped reference image from LC during train-
ing.
Method
SSIM↑
LPIPS↓
FID↓
w/o LiN
0.7115
0.3466
207.33
w/o LiLPIPS
0.8015
0.3237
160.77
w/o LiC
0.8208
0.2209
113.95
Ours
0.7913
0.2267
104.00
lower scores in LPIPS and SSIM compared to the w/o LiC set-
ting, Fig. 10 demonstrates that using all loss terms produces
the most qualitatively satisfying results. Without normal loss,
the geometry becomes less accurate, resulting in distortions
that fall within the acceptable threshold of the geometry consis-
tency measure. Without a perceptual loss to guide the integra-
tion from inconsistent multi-view inpainting inputs, appearance
tends to degrade where depth and normal information alone is
11
Figure 7: Visual comparison of test set renderings. This figure presents sample renderings from the novel test view: (a)–(c) are from SPIn-NeRF, (d)–(e) from
MipNeRF360, and (f)–(h) from our custom dataset. Each scene is shown in two rows—the top row presents the full image, while the bottom row shows a cropped
region focusing on the inpainted area. Arrows indicate areas where the baseline underperforms. Our method not only preserves fine details from the inpainting input
but also reconstructs perceptually consistent scenes across views, guided by a single reference image. For clearer comparison, some images have been adjusted in
brightness and contrast.
Table 4: An ablation study on the effect of confidence-guided view weight-
ing was conducted on our custom dataset. Uniform-Weight uses equal weights
for all views, Confidence-Threshold excludes low-confidence views based on a
threshold, and Confidence-Weighted adaptively weights views by confidence.
Method
SSIM↑
LPIPS↓
FID↓
Uniform-Weight
0.7763
0.2312
107.93
Confidence-Threshold
0.7889
0.2278
104.28
Confidence-Weighted (ours)
0.7913
0.2267
104.00
(Eq. (12)). Specifically, we sequentially exclude the normal,
perceptual, and color loss on the warped reference image. The
depth loss, which is commonly used in prior work, is retained
as its effectiveness has already been well established. Here, w/o
LiC denotes the removal of the inpainting color loss component
from LC, specifically by excluding the warped reference image
from the calculation. Although LC is not strictly decomposable,
we use it for clarity in the ablation setting.
Although the final proposed method seems to show slightly
Table 5: Ablation study of our proposed loss formulation on the custom dataset.
We evaluate the contribution of each loss term by removing them individually.
w/o LiC refers to excluding the warped reference image from LC during train-
ing.
Method
SSIM↑
LPIPS↓
FID↓
w/o LiN
0.7115
0.3466
207.33
w/o LiLPIPS
0.8015
0.3237
160.77
w/o LiC
0.8208
0.2209
113.95
Ours
0.7913
0.2267
104.00
lower scores in LPIPS and SSIM compared to the w/o LiC set-
ting, Fig. 10 demonstrates that using all loss terms produces
the most qualitatively satisfying results. Without normal loss,
the geometry becomes less accurate, resulting in distortions
that fall within the acceptable threshold of the geometry consis-
tency measure. Without a perceptual loss to guide the integra-
tion from inconsistent multi-view inpainting inputs, appearance
tends to degrade where depth and normal information alone is
11

<!-- page 12 -->
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
SPIn-NeRF (reimpl.)
Gaussian Grouping
Infusion
Ours
Figure 8: Qualitative comparison on three novel views. This figure demonstrates the effectiveness of our method in producing coherent and photorealistic 3D scene
inpainting. Arrows indicate areas where the baseline underperforms. For clearer visualization, the images are cropped to the inpainted regions, with brightness and
contrast adjusted where necessary.
12

<!-- page 13 -->
Reference image w/ Original
Novel view #1
Novel view #2
Novel view #3
Figure 9: High-fidelity appearance from a single reference view. We present sample novel views from wide-view datasets (MipNeRF360 [2] and our custom scenes),
demonstrating that our method produces results that remain perceptually faithful to the reference across a wide range of viewpoints. The images show cropped
regions of the inpainted areas, with brightness adjusted for better inspection.
insufficient to reconstruct accurate geometry. Furthermore, re-
lying on the appearance reconstruction solely on LPIPS loss
without the support of L1 and SSIM losses leads to a noticeable
degradation in fine details.
4.5. Limitations
While our method enables more efficient and stable 3D scene
inpainting than approaches relying on reprojection from limited
views, some dependency on depth estimation remains. It may
suffer performance degradation in scenes with poor geometric
consistency. As shown in Fig. 11, slight hallucinations appear
around edge regions with abrupt depth changes, and distortions
can be observed from view directions where high-confidence
inpainting inputs were insufficient. While our method relaxes
the implicit assumption in previous approaches that inpainting
across views generally shares perceptual similarity, it still relies
on geometric consistency. Therefore, it is vulnerable in situa-
tions where obtaining geometrically consistent inpaintings be-
comes difficult due to highly irregular or complex scenes, such
as forests. To achieve robust 3D inpainting in more complex
scenes, new methods that do not depend on inpainting consis-
tency are required.
Additionally,
although we apply Poisson blending for
warping to achieve seamless integration and preserve view-
dependent effects, its effectiveness diminishes when the inpaint-
ing region spans multiple objects or when residual shadows of
the removed object remain.
5. Conclusion
In this paper, we propose a 3D scene inpainting method on
3DGS that reconstructs a coherent 3D scene. Based on a single
user-specified reference view, our method estimates the relia-
bility of inpainted results from other views by measuring their
similarity to the reference. By leveraging all high-confidence
information, it constructs a stable geometry and achieves a
perceptually consistent appearance across multiple viewpoints
13
Figure 9: High-fidelity appearance from a single reference view. We present sample novel views from wide-view datasets (MipNeRF360 [2] and our custom scenes),
demonstrating that our method produces results that remain perceptually faithful to the reference across a wide range of viewpoints. The images show cropped
regions of the inpainted areas, with brightness adjusted for better inspection.
insufficient to reconstruct accurate geometry. Furthermore, re-
lying on the appearance reconstruction solely on LPIPS loss
without the support of L1 and SSIM losses leads to a noticeable
degradation in fine details.
4.5. Limitations
While our method enables more efficient and stable 3D scene
inpainting than approaches relying on reprojection from limited
views, some dependency on depth estimation remains. It may
suffer performance degradation in scenes with poor geometric
consistency. As shown in Fig. 11, slight hallucinations appear
around edge regions with abrupt depth changes, and distortions
can be observed from view directions where high-confidence
inpainting inputs were insufficient. While our method relaxes
the implicit assumption in previous approaches that inpainting
across views generally shares perceptual similarity, it still relies
on geometric consistency. Therefore, it is vulnerable in situa-
tions where obtaining geometrically consistent inpaintings be-
comes difficult due to highly irregular or complex scenes, such
as forests. To achieve robust 3D inpainting in more complex
scenes, new methods that do not depend on inpainting consis-
tency are required.
Additionally,
although we apply Poisson blending for
warping to achieve seamless integration and preserve view-
dependent effects, its effectiveness diminishes when the inpaint-
ing region spans multiple objects or when residual shadows of
the removed object remain.
5. Conclusion
In this paper, we propose a 3D scene inpainting method on
3DGS that reconstructs a coherent 3D scene. Based on a single
user-specified reference view, our method estimates the relia-
bility of inpainted results from other views by measuring their
similarity to the reference. By leveraging all high-confidence
information, it constructs a stable geometry and achieves a
13

<!-- page 14 -->
Ground truth
w/o LiN
w/o LiLPIPS
w/o LiC
Ours
Figure 10: Qualitative examples from the ablation study on loss term combinations. The figure shows sample results on our custom dataset. Excluding normal,
LPIPS, or color loss in the inpainted region hinders geometric consistency or reduces appearance fidelity and structural detail.
(a) Hallucination
(b) Distortion
Figure 11: Example of degraded performance. Due to the reliance on inpainted
depth, hallucinations may appear at boundaries with abrupt depth changes
in cases of severe inconsistency. For view directions with insufficient high-
confidence inpainting, the reconstructed shape can be less accurate.
perceptually consistent appearance across multiple viewpoints
through warping. Experimental results demonstrate the effec-
tiveness of our proposed method over existing 3D scene inpaint-
ing methods.
For future work, we aim to reduce the reliance on the geomet-
rical consistency of inpainted images by leveraging personal-
ized diffusion models [48, 49] to generate viewpoint-consistent
inpaintings that encourage visual similarity to the reference im-
age. Furthermore, we plan to investigate approaches for esti-
mating uncertainty within the scene to enhance the identifica-
tion of regions that require inpainting in 3D Gaussian scenes
following object removal.
Acknowledgments
This work was supported by the Industrial Technology Inno-
vation Program (20012462) funded by the Ministry of Trade,
Industry & Energy (MOTIE, Korea), the National Research
Foundation of Korea (NRF) grant (NRF-2021R1A2C2093065)
funded by the Korea government (MSIT) and the KIST under
the Institutional Program (Grant No. 2E33841).
References
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
R. Ng, Nerf: Representing scenes as neural radiance fields for view syn-
thesis, Commun. ACM 65 (1) (2021) 99–106. doi:10.1145/3503250.
URL https://doi.org/10.1145/3503250
[2] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman, Mip-
nerf 360: Unbounded anti-aliased neural radiance fields, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2022, pp. 5470–5479. doi:10.1109/CVPR52688.2022.
00539.
URL https://doi.org/10.1109/CVPR52688.2022.00539
[3] T. M¨uller, A. Evans, C. Schied, A. Keller, Instant neural graphics prim-
itives with a multiresolution hash encoding, ACM Trans. Graph. 41 (4)
(2022) 102:1–102:15. doi:10.1145/3528223.3530127.
URL https://doi.org/10.1145/3528223.3530127
[4] B. Kerbl, G. Kopanas, T. Leimkuehler, G. Drettakis, 3d gaussian splat-
ting for real-time radiance field rendering, ACM Trans. Graph. 42 (4) (jul
2023). doi:10.1145/3592433.
URL https://doi.org/10.1145/3592433
[5] A. Gu´edon, V. Lepetit, Sugar: Surface-aligned gaussian splatting for
efficient 3d mesh reconstruction and high-quality mesh rendering, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
14

<!-- page 15 -->
Pattern Recognition (CVPR), 2024, pp. 5354–5363.
doi:10.1109/
CVPR52733.2024.00512.
URL https://doi.org/10.1109/CVPR52733.2024.00512
[6] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, X. Chen,
Gaussianpro: 3d gaussian splatting with progressive propagation, in:
arXiv preprint arXiv:2402.14650, 2024. doi:10.48550/arXiv.2402.
14650.
URL https://doi.org/10.48550/arXiv.2402.14650
[7] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai,
L. Yang, H. Liu, G. Lin, Gaussianeditor: Swift and controllable 3d edit-
ing with gaussian splatting, in: 2024 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2024, pp. 21476–21485.
doi:10.1109/CVPR52733.2024.02029.
URL https://doi.org/10.1109/CVPR52733.2024.02029
[8] V. Rudnev, M. Elgharib, W. Smith, L. Liu, V. Golyanik, C. Theobalt,
Nerf for outdoor scene relighting, in: Computer Vision – ECCV 2022,
Springer Nature Switzerland, Cham, 2022, pp. 615–631. doi:10.1007/
978-3-031-19787-1{\_}35.
URL https://doi.org/10.1007/978-3-031-19787-1{_}35
[9] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, K. Jia,
GS-IR: 3D Gaussian
Splatting for Inverse Rendering , in: 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), IEEE Computer So-
ciety, Los Alamitos, CA, USA, 2024, pp. 21644–21653. doi:10.1109/
CVPR52733.2024.02045.
URL https://doi.org/10.1109/CVPR52733.2024.02045
[10] J. Gao, C. Gu, Y. Lin, Z. Li, H. Zhu, X. Cao, L. Zhang, Y. Yao,
Relightable 3d gaussians:
Realistic point cloud relighting with brdf
decomposition and ray tracing, in: Computer Vision – ECCV 2024,
Springer Nature Switzerland, Cham, 2025, pp. 73–89. doi:10.1007/
978-3-031-72995-9{\_}5.
URL https://doi.org/10.1007/978-3-031-72995-9{_}5
[11] K. Liu, F. Zhan, Y. Chen, J. Zhang, Y. Yu, A. El Saddik, S. Lu, E. P.
Xing, Stylerf: Zero-shot 3d style transfer of neural radiance fields, in:
2023 IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2023, pp. 8338–8348. doi:10.1109/CVPR52729.2023.
00806.
URL https://doi.org/10.1109/CVPR52729.2023.00806
[12] K. Liu, F. Zhan, M. Xu, C. Theobalt, L. Shao, S. Lu, Stylegaussian: In-
stant 3d style transfer with gaussian splatting, in: SIGGRAPH Asia 2024
Technical Communications, SA ’24, Association for Computing Machin-
ery, New York, NY, USA, 2024. doi:10.1145/3681758.3698002.
URL https://doi.org/10.1145/3681758.3698002
[13] B. Yang, Y. Zhang, Y. Xu, Y. Li, H. Zhou, H. Bao, G. Zhang, Z. Cui,
Learning object-compositional neural radiance field for editable scene
rendering, in: 2021 IEEE/CVF International Conference on Computer
Vision (ICCV), IEEE Computer Society, Los Alamitos, CA, USA, 2021,
pp. 13759–13768. doi:10.1109/ICCV48922.2021.01352.
URL https://doi.org/10.1109/ICCV48922.2021.01352
[14] T. Anciukeviˇcius, Z. Xu, M. Fisher, P. Henderson, H. Bilen, N. J. Mitra,
P. Guerrero, Renderdiffusion: Image diffusion for 3d reconstruction, in-
painting and generation, in: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 12608–
12618. doi:10.1109/CVPR52729.2023.01213.
URL https://doi.org/10.1109/CVPR52729.2023.01213
[15] J. Xiang, J. Yang, B. Huang, X. Tong, 3d-aware image generation us-
ing 2d diffusion models, in: Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV), 2023, pp. 2383–2393.
doi:10.1109/ICCV51070.2023.00226.
URL https://doi.org/10.1109/ICCV51070.2023.00226
[16] E. Chan, M. Monteiro, P. Kellnhofer, J. Wu, G. Wetzstein, pi-gan: Peri-
odic implicit generative adversarial networks for 3d-aware image synthe-
sis, in: arXiv, 2020. doi:10.1109/CVPR46437.2021.00574.
URL https://doi.org/10.1109/CVPR46437.2021.00574
[17] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha,
A. Silvestrov, N. Kong, H. Goka, K. Park, V. Lempitsky, Resolution-
robust large mask inpainting with fourier convolutions, in: Proceedings
of the IEEE/CVF Winter Conference on Applications of Computer Vi-
sion (WACV), 2022, pp. 2149–2159. doi:10.1109/WACV51458.2022.
00323.
URL https://doi.org/10.1109/WACV51458.2022.00323
[18] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, High-
resolution image synthesis with latent diffusion models, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2022, pp. 10684–10695.
doi:10.1109/CVPR52688.
2022.01042.
URL https://doi.org/10.1109/CVPR52688.2022.01042
[19] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller,
J. Penna, R. Rombach, Sdxl: Improving latent diffusion models for
high-resolution image synthesis (2023).
arXiv:2307.01952, doi:
10.48550/arXiv.2307.01952.
URL https://doi.org/10.48550/arXiv.2307.01952
[20] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A.
Brubaker, I. Gilitschenski, A. Levinshtein, Spin-nerf: Multiview seg-
mentation and perceptual inpainting with neural radiance fields, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), 2023, pp. 20669–20679.
doi:10.1109/
CVPR52729.2023.01980.
URL https://doi.org/10.1109/CVPR52729.2023.01980
[21] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The unreason-
able effectiveness of deep features as a perceptual metric, in: Proceed-
ings of the IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2018. doi:10.1109/CVPR.2018.00068.
URL https://doi.org/10.1109/CVPR.2018.00068
[22] Y. Yin, Z. Fu, F. Yang, G. Lin, Or-nerf: Object removing from 3d scenes
guided by multiview segmentation with neural radiance fields (2023).
arXiv:2305.10503, doi:10.48550/arXiv.2305.10503.
URL https://doi.org/10.48550/arXiv.2305.10503
[23] J. Huang, H. Yu, J. Zhang, H. Nait-Charif, Point’n move:
In-
teractive scene object manipulation on gaussian splatting radiance
fields (2024).
arXiv:https://digital-library.theiet.org/
doi/pdf/10.1049/ipr2.13190, doi:10.1049/ipr2.13190.
URL https://doi.org/10.1049/ipr2.13190
[24] D. Wang, T. Zhang, A. Abboud, S. S¨usstrunk, Inpaintnerf360: Text-
guided 3d inpainting on unbounded neural radiance fields (2023). arXiv:
2305.15094, doi:10.48550/arXiv.2305.15094.
URL https://doi.org/10.48550/arXiv.2305.15094
[25] M. Ye, M. Danelljan, F. Yu, L. Ke, Gaussian grouping:
Segment
and edit anything in 3d scenes, in: Computer Vision - ECCV 2024,
Springer-Verlag, Berlin, Heidelberg, 2024, p. 162–179. doi:10.1007/
978-3-031-73397-0{\_}10.
URL https://doi.org/10.1007/978-3-031-73397-0{_}10
[26] A. Mirzaei, T. Aumentado-Armstrong, M. A. Brubaker, J. Kelly,
A. Levinshtein, K. G. Derpanis, I. Gilitschenski, Reference-guided con-
trollable inpainting of neural radiance fields, in: 2023 IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), IEEE Computer Soci-
ety, Los Alamitos, CA, USA, 2023, pp. 17769–17779. doi:10.1109/
ICCV51070.2023.01633.
URL https://doi.org/10.1109/ICCV51070.2023.01633
[27] I.-C. Shen, H.-K. Liu, B.-Y. Chen, Nerf-in: Free-form inpainting for pre-
trained nerf with rgb-d priors, IEEE Computer Graphics and Applications
44 (2) (2024) 100–109. doi:10.1109/MCG.2023.3336224.
[28] S. Weder, G. Garcia-Hernando, A. Monszpart, M. Pollefeys, G. J. Bros-
tow, M. Firman, S. Vicente, Removing objects from neural radiance
fields, in: Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR), 2023, pp. 16528–16538. doi:
10.1109/CVPR52729.2023.01586.
URL https://doi.org/10.1109/CVPR52729.2023.01586
[29] Z. Liu, H. Ouyang, Q. Wang, K. L. Cheng, J. Xiao, K. Zhu, N. Xue,
Y. Liu, Y. Shen, Y. Cao, Infusion: Inpainting 3d gaussians via learn-
ing depth completion from diffusion prior (2024). arXiv:2404.11613,
doi:10.48550/arXiv.2404.11613.
URL https://doi.org/10.48550/arXiv.2404.11613
[30] R. Fischer, J. Roßkamp, T. Hudcovic, A. Schlegel, G. Zachmann, Inpaint-
ing of depth images using deep neural networks for real-time applica-
tions, in: G. Bebis, G. Ghiasi, Y. Fang, A. Sharf, Y. Dong, C. Weaver,
Z. Leo, J. J. LaViola Jr., L. Kohli (Eds.), Advances in Visual Com-
puting, Springer Nature Switzerland, Cham, 2023, pp. 121–135. doi:
10.1007/978-3-031-47966-3{\_}10.
URL https://doi.org/10.1007/978-3-031-47966-3{_}10
[31] D. Wang, T. Zhang, A. Abboud, S. S¨usstrunk, Innerf360: Text-guided 3d-
consistent object inpainting on 360-degree neural radiance fields, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
15

<!-- page 16 -->
Recognition (CVPR), 2024. doi:10.1109/CVPR52733.2024.01205.
URL https://doi.org/10.1109/CVPR52733.2024.01205
[32] K. Deng, A. Liu, J.-Y. Zhu, D. Ramanan, Depth-supervised nerf: Fewer
views and faster training for free, in: 2022 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12872–
12881. doi:10.1109/CVPR52688.2022.01254.
[33] J. Chung, J. Oh, K. M. Lee, Depth-regularized optimization for 3d gaus-
sian splatting in few-shot images, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) Work-
shops, 2024, pp. 811–820. doi:10.1109/CVPRW63382.2024.00086.
URL https://doi.org/10.1109/CVPRW63382.2024.00086
[34] M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, J. Kannala,
Dn-splatter: Depth and normal priors for gaussian splatting and meshing,
in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vi-
sion (WACV), 2025, pp. 2421–2431. doi:10.1109/WACV61041.2025.
00241.
URL https://doi.org/10.1109/WACV61041.2025.00241
[35] Z. Zhu, Z. Fan, Y. Jiang, Z. Wang, Fsgs: Real-time few-shot view syn-
thesis using gaussian splatting, in: Computer Vision – ECCV 2024: 18th
European Conference, Milan, Italy, September 29–October 4, 2024, Pro-
ceedings, Part XXXIX, Springer-Verlag, Berlin, Heidelberg, 2024, p.
145–163. doi:10.1007/978-3-031-72933-1{\_}9.
URL https://doi.org/10.1007/978-3-031-72933-1{_}9
[36] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, M. Nießner,
Dense depth priors for neural radiance fields from sparse input views,
in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2022, pp. 12882–12891.
doi:10.1109/CVPR52688.
2022.01255.
URL https://doi.org/10.1109/CVPR52688.2022.01255
[37] J. L. Sch¨onberger, J.-M. Frahm, Structure-from-motion revisited, in: 2016
IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2016, pp. 4104–4113. doi:10.1109/CVPR.2016.445.
URL https://doi.org/10.1109/CVPR.2016.445
[38] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr,
R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala,
N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, C. Feichtenhofer, Sam 2:
Segment anything in images and videos (2024). arXiv:2408.00714,
doi:10.48550/arXiv.2408.00714.
URL https://arxiv.org/abs/2408.00714
[39] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao, Depth
anything v2, arXiv:2406.09414 (2024). doi:10.48550/arXiv.2406.
09414.
URL https://doi.org/10.48550/arXiv.2406.09414
[40] C. Tomasi, R. Manduchi, Bilateral filtering for gray and color im-
ages, in: Sixth International Conference on Computer Vision (IEEE
Cat. No.98CH36271), 1998, pp. 839–846. doi:10.1109/ICCV.1998.
710815.
[41] M.-S. Kwak, J. Song, S. Kim, Geconerf: few-shot neural radiance fields
via geometric consistency, in: Proceedings of the 40th International Con-
ference on Machine Learning (ICML), 2023.
URL https://dl.acm.org/doi/10.5555/3618408.3619152
[42] P. P´erez, M. Gangnet, A. Blake, Poisson image editing, in: ACM SIG-
GRAPH 2003 Papers, SIGGRAPH ’03, Association for Computing Ma-
chinery, New York, NY, USA, 2003, p. 313–318.
doi:10.1145/
1201775.882269.
URL https://doi.org/10.1145/1201775.882269
[43] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, High-
resolution image synthesis with latent diffusion models, in:
2022
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2022, pp. 10674–10685.
doi:10.1109/CVPR52688.2022.
01042.
URL https://doi.org/10.1109/CVPR52688.2022.01042
[44] Blender Online Community, Blender 3.5.1 - a 3D modelling and render-
ing package, Blender Foundation, Amsterdam, The Netherlands (2023).
URL https://www.blender.org
[45] BlenderKit,
Online
asset
library
for
blender,
https://www.
blenderkit.com, accessed: 2025-04.
[46] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter, Gans
trained by a two time-scale update rule converge to a local nash equi-
librium, in: Proceedings of the 31st International Conference on Neural
Information Processing Systems, NIPS’17, Curran Associates Inc., Red
Hook, NY, USA, 2017, p. 6629–6640.
URL https://dl.acm.org/doi/10.5555/3295222.3295408
[47] Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, Image quality assessment:
from error visibility to structural similarity, IEEE Transactions on Image
Processing 13 (4) (2004) 600–612. doi:10.1109/TIP.2003.819861.
[48] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, K. Aberman, Dream-
booth: Fine tuning text-to-image diffusion models for subject-driven gen-
eration (2023) 22500–22510doi:10.1109/CVPR52729.2023.02155.
URL https://doi.org/10.1109/CVPR52729.2023.02155
[49] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, J.-Y. Zhu, Multi-concept
customization of text-to-image diffusion, in: 2023 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1931–
1941. doi:10.1109/CVPR52729.2023.00192.
URL https://doi.org/10.1109/CVPR52729.2023.00192
16
