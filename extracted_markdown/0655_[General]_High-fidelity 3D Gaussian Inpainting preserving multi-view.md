<!-- page 1 -->
1
High-fidelity 3D Gaussian Inpainting: preserving
multi-view consistency and photorealistic details
Jun Zhou, Dinghao Li, Nannan Li, Mingjie Wang
Abstract—Recent advancements in multi-view 3D reconstruc-
tion and novel-view synthesis, particularly through Neural Ra-
diance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have
greatly enhanced the fidelity and efficiency of 3D content creation.
However, inpainting 3D scenes remains a challenging task due to
the inherent irregularity of 3D structures and the critical need
for maintaining multi-view consistency. In this work, we propose
a novel 3D Gaussian inpainting framework that reconstructs
complete 3D scenes by leveraging sparse inpainted views. Our
framework incorporates an automatic Mask Refinement Process
and region-wise Uncertainty-guided Optimization. Specifically,
we refine the inpainting mask using a series of operations, includ-
ing Gaussian scene filtering and back-projection, enabling more
accurate localization of occluded regions and realistic boundary
restoration. Furthermore, our Uncertainty-guided Fine-grained
Optimization strategy, which estimates the importance of each
region across multi-view images during training, alleviates multi-
view inconsistencies and enhances the fidelity of fine details
in the inpainted results. Comprehensive experiments conducted
on diverse datasets demonstrate that our approach outperforms
existing state-of-the-art methods in both visual quality and view
consistency.
Index Terms—3D Gaussian Splatting, 3D Scene Inpainting,
Automatic Mask Refinement, Multi-view Consistency.
I. INTRODUCTION
Multi-view 3D reconstruction and novel-view synthesis are
crucial for creating high-fidelity 3D content of real-world
scenes, enabling applications such as telepresence and AR/VR.
Recent advancements in Neural Radiance Fields (NeRF)[1,
2, 3] and 3D Gaussian Splatting (3DGS)[4, 5, 6, 7] have
significantly accelerated progress in this field. Among these,
3D Gaussian-based approaches have garnered substantial at-
tention due to their ability to produce photorealistic images
while achieving impressive rendering speeds. By leveraging
the advantages of 3D Gaussian Splatting, researchers can more
easily construct and generate richer 3D [8, 9, 9] and even 4D
assets [10, 11, 12], as well as reconstruct physical laws [13, 14]
from scenes to enable interactive engagement [15]. As a result,
the demand for technologies that facilitate the editing and
manipulation of such scenes has surged. Among these, 3D
scene inpainting has emerged as a prominent research focus.
The authors would like to thank the High Performance Computing Center
of Dalian Maritime University for providing the computing resources. This
work was supported by NSFC (No. 62002040) and Fundamental Research
Funds for the Central Universities (No. 3132025274).(Corresponding author:
Jun Zhou.)
J. Zhou, D. Li, and N. Li are with the School of Information Sci-
ence and Technology, Dalian Maritime University, Dalian, China (E-mail:
jun90@dlmu.edu.cn, ldh123@dlmu.edu.cn, nannanli@dlmu.edu.cn).
M. Wang is with the School of Science, Zhejiang Sci-Tech University, Zhe
Jiang, China (E-mail: mingjiew@zstu.edu.cn).
While inpainting techniques have been extensively studied in
2D image domains [16], the challenge of inpainting 3D scenes
remains significant due to the multi-view nature of scene data
and the irregular structures inherent in 3D representations.
These complexities make 3D scene inpainting both a demand-
ing and promising area of exploration.
As pioneering works, Remove-NeRF [20] and SPIn-
NeRF [21] have demonstrated object removal and scene in-
painting using NeRF representations [1]. These methods intro-
duced a 2D-to-3D strategy for 3D scene inpainting, utilizing
LaMa [17] for object removal and inpainting across multi-view
images, followed by optimizing the 3D scene with a view-
consistency constraint. However, the LaMa-based inpainting
technique tends to introduce blurriness in the inpainted images
and lacks fine image details, while NeRF-based optimization
requires considerable time, resulting in reduced temporal effi-
ciency.
In recent years, driven by the efficiency advantages of 3D
Gaussian representations [4], a variety of 3D scene editing
and inpainting techniques [22, 23, 24, 25, 19] based on Gaus-
sian representations have been extensively explored. Among
these, InFusion [19] stands out, leveraging a latent diffusion
model [18] to simultaneously inpaint both RGB and depth
images, achieving richer image details. While the diffusion
model enhances detail restoration compared to LaMa, it in-
troduces increased multi-view inconsistencies. As illustrated
in Fig.1, we select four sparse keyframes with distinct view-
points for 3DGS Inpainting. The results show that LaMa-
based inpainting produces blurrier images with less realistic
detail compared to diffusion model. However, LaMa ensures
better multi-view consistency, helping to avoid inconsistencies
in 3DGS Inpainting at the cost of detail richness, which
results in less realistic outcomes (Fig.1 (a), second row on
the right). In contrast, diffusion-based techniques offer richer
details but introduce significant multi-view inconsistencies.
When trained with multiple viewpoints, these inconsistencies
are exacerbated, leading to blurry results (Fig. 1 (a), first row
on the right).
To address the issue of multi-view inconsistencies in 3DGS
Inpainting, we explored two optimization strategies: single-
view supervision and the InFusion method, which jointly op-
timizes RGB and depth images, as illustrated in Fig. 1 (b) and
(c), second row. In both cases, training was conducted solely
using the image from viewpoint V4. While these approaches
produced sharp results for the supervised view, they often led
to blurriness or missing content in other views, particularly in
complex, large-scale scenes. This performance degradation is
primarily due to the limited information provided by single-
arXiv:2507.18023v1  [cs.CV]  24 Jul 2025

<!-- page 2 -->
2
Fig. 1.
Comparison of Different Optimization Strategies for 3D Gaussian Inpainting: (1) The left section shows the four keyframes used for training; (2)
the middle section compares inpainted keyframes using LaMa [17] and a diffusion-based method [18]; (3) the right section presents the final 3D Gaussian
scene inpainting results from seven viewpoints, including keyframes and intermediate views. Specifically, (a) compares the results using all keyframes with
diffusion-based (first row) and LaMa-based (second row) methods; (b) compares progressive and single-view (view V4) strategies; (c) showcases results from
InFusion [19] using both progressive and single-view (view V4) approaches; and (d) highlights the results of our proposed method, which strikes a better
balance between multi-view consistency and the preservation of realistic scene details.
view supervision, which causes overfitting to the supervised
viewpoint. We further experimented with a progressive training
strategy, shown in the first rows of (b) and (c), where additional
views were introduced over time. However, whether using
basic image supervision or InFusion’s progressive method,
earlier view information tended to be forgotten in later stages,
resulting in blurred transitional views and unsuccessful scene
inpainting. To overcome these limitations, we propose a
novel Uncertainty-guided Fine-grained Optimization strategy
by depth-based view selection. By selectively extracting infor-
mative regions across sparse viewpoints, our method achieves
a better balance between multi-view consistency and realistic
detail preservation. As shown in Fig.1 (d), our approach
yields more coherent and visually convincing results. While
conceptually similar to the multi-view selection strategy in
Remove-NeRF[20], our method differs in two key aspects:
(1) it relies on sparse supervision from a few key views, and
(2) it performs fine-grained region-wise selection based on
depth, where regions closer to the camera are assigned higher
confidence due to their greater reliability.
Furthermore, to enhance both the efficiency and accuracy of
3DGS Inpainting, we designed a more precise automatic Mask
Refinement algorithm. Specifically, we propose a mask opti-
mization strategy that adjusts and refines the initial segmenta-
tion mask to minimize its size while preserving occluded real
scene information more accurately. This refinement provides
a solid and accurate data foundation for subsequent inpainting
processes, significantly improving the precision and realism of
the results.
In summary, our main contributions are as follows:
We present a novel 3DGS Inpainting framework specifically
designed for sparse-view inputs. This framework integrates
an automatic Mask Refinement Process that extracts addi-
tional effective background information, along with a depth-
based Uncertainty-guided Fine-grained Optimization strategy
that strikes a balance between multi-view consistency and
the preservation of rich visual details. Extensive experiments
conducted on multiple benchmark datasets demonstrate that
our method outperforms existing state-of-the-art approaches
in the 3DGS Inpainting task.
II. RELATED WORK
A. Image and Video Inpainting
In the field of computer vision [26], video and image
inpainting aims to restore missing regions while ensuring
seamless blending and realistic details. Early methods [27, 28,
29, 30, 31, 32, 33, 34, 35] relied on local background cues
and optimization techniques, while traditional video inpainting
approaches [36, 37, 38, 39] extended these ideas to handle
temporal consistency. However, they often failed in cases
with large missing areas or complex motion. Recently, deep
learning methods, especially those based on transformers and
diffusion models [17, 18], have achieved impressive results,
effectively overcoming these limitations. In this work, we
adopt these methods to generate multi-view inpainting results,
taking advantage of its ability to produce highly realistic
and visually coherent images. However, compared to conven-
tional image or video inpainting, 3D scene inpainting presents
additional challenges. One of the most critical difficulties
lies in maintaining consistency across multiple views while
simultaneously preserving fine-grained realism from different
viewpoints.
B. Radiance Fields and Rendering
Photorealistic view synthesis is a long-standing challenge
in computer vision and computer graphics. Traditional 3D

<!-- page 3 -->
3
representations such as meshes and point clouds remain widely
used due to their explicit geometry and efficient GPU-based
rasterization. In recent years, Neural Fields have emerged
as a powerful alternative, offering seamless integration with
deep learning frameworks and enabling high-quality novel
view synthesis. Neural Fields can generally be divided into
three main types. Early methods [1, 2, 40] model radiance
fields with MLPs for high-quality view synthesis, but suffer
from slow rendering due to dense ray sampling. Acceleration
techniques alleviate this but often increase memory usage or
degrade visual fidelity. Then, the grid-based methods [41, 42]
discretize space into voxel or hash grids to enable fast
interpolation-based rendering, offering efficiency gains but
still requiring many samples and struggling with empty space
representation. Building on 3DGS, a series of follow-up works,
including 2DGS [43] and Scaffold-GS [7], have introduced
further enhancements. Our work is also based on the 3DGS
representation.
C. Radiance Fields Inpainting
In recent years, 3D Radiance Fields have gradually emerged
as a novel 3D representation, driving an increasing demand
for 3D editing [23, 24, 44, 45, 46, 22]. These techniques
support a variety of scene editing operations, from object
replacement to appearance adjustments, granting users im-
proved control and flexibility in 3D scene manipulation. 3D
Radiance Fields Inpainting is one prominent application that
can restore missing regions in 3D scenes, ensuring high-quality
and consistent multi-view rendering results. Notably, although
the aforementioned editing techniques mention inpainting in
3D scenes, they primarily treat it as a post-processing step by
applying image inpainting methods to the removed regions.
This approach does not perform genuine inpainting directly
on the 3D scene, leaving the consistency and integrity of the
reconstructed scene unaddressed.
As pioneering efforts, NeRF-In [47] and SPIn-NeRF [21]
utilize multi-view images to restore NeRF representations.
However, they fail to address the multi-view consistency issues
that arise from discrepancies in the inpainted regions. To tackle
this, View-Substitute [48] proposes inpainting a single refer-
ence view and guiding the synthesis of other views via depth
warping and bilateral filtering to ensure consistency. Neverthe-
less, its reliance on a single-view reference limits performance
when dealing with complex or large missing regions. Sub-
sequent works like Removal-NeRF [20] enhance consistency
through confidence-based view selection, while OR-NeRF [49]
introduces efficient multi-view segmentation combined with
an integrated TensoRF framework to achieve higher-quality
rendering. Although these techniques have achieved notable
progress, the emergence of 3DGS has highlighted the growing
demand for faster inpainting methods leveraging point-based
rendering techniques.
Among these 3DGS inpainting methods, InFusion [19] em-
ploys a depth-generative diffusion model to synthesize RGB-
D point clouds, which are then fused with the missing 3DGS
regions for inpainting. However, this method still struggles
to effectively balance high fidelity and multi-view consis-
tency. Lu et al.[50] proposed a technique similar to View-
Substitute, focusing on repairing a single keyframe and using
depth projection to construct consistent data for other views.
Yet, it still faces difficulties in handling complex and large-
scale missing regions. Wang et al.[51] used Scaffold-GS [7]
as the backbone and introduced an attention mechanism to
learn consistent Gaussian features for the missing regions.
However, multi-view inconsistencies remain unresolved. Sim-
ilarly, Point’n Move [52] leveraged 2D prompt points as
interactive inputs to identify missing regions and adopted a
”minimize changes” optimization strategy, akin to our mask
refinement approach. Despite these efforts, it still falls short
of addressing multi-view inconsistencies, limiting its ability
to effectively inpaint complex and large missing areas. The
method by Gaussian Group [24] focuses more on the semantic
segmentation of objects within the Gaussian scene. Incorrect
semantics can lead to inaccurate mask estimation, resulting in
suboptimal inpainting. Concurrently, Huang et al. [53] propose
a depth-guided multi-view strategy for consistent inpainting.
Although their multi-view mask warping yields finer masks,
it overlooks rendering instability near mask edges, leading
to overly smooth results. Our work tackles these challenges
by emphasizing multi-view consistency and the accuracy of
mask estimation, ultimately improving the quality of 3DGS
inpainting results.
III. METHOD
A. Overview
Our work aims to reconstruct and inpaint 3D scenes from
multi-view images and masks, achieving consistent and photo-
realistic representations. Built upon 3D Gaussian Splatting [4],
we extend its capabilities to address 3D scene inpainting. Sim-
ilar to methods like InFusion [19] and Gaussian Group [24],
our approach adopts a two-stage pipeline. In the first stage,
we reconstruct a 3D scene with ”holes” by utilizing masks
and multi-view inputs, recovering regions outside the missing
areas by leveraging background information from other views.
A brief overview of this step is provided in the preliminary
section (Sec.III-B). In the second stage, we inpaint the missing
content within the ”holes” using image inpainting techniques.
To improve this process, we introduce an automatic Mask
Refinement method (Sec.III-C) for more accurate hole defini-
tion and propose a novel training framework (Sec.III-D) that
incorporates depth-based uncertainty scores to balance multi-
view consistency and fine-detail preservation. An overview of
our full pipeline is illustrated in Fig.2.
B. Preliminaries: 3D Gaussian Scene Initialization with
Masks
The Gaussian Splitting [4] can be used to reconstruct
3D Gaussian representation from multi-view images. The
Gaussian representation inherently possess rich geometric at-
tributes, and can also be employed for rendering new view
synthesis. Similar to Infusion [19], we need to utilize both
multi-view images Co = {co
i }N
i=1, accompanied by respective
camera poses Π = {πi}N
i=1 and their corresponding masks
M = {mi}N
i=1 for scene reconstruction, with the requirement

<!-- page 4 -->
4
Fig. 2. Overview of our 3D Gaussian inpainting pipeline with Mask Refinement Process and Uncertainty-guided Optimization. Given a set of posed input
images and their coarse binary masks, we first perform an initial training of the 3D Gaussian scene representation. Based on this initial representation,
we introduce an automatic Mask Refinement module that accurately localizes regions requiring inpainting. In the second stage, we perform Uncertainty-
guided Optimization, which selectively utilizes reliable supervision from inpainted images. This strategy effectively mitigates conflicts arising from multi-view
inconsistencies and leads to a more coherent and photo-realistic 3D scene synthesis.
to remove the masked regions. Specifically, our objective is
to train an initial 3D Gaussian representation Θ = {gi}L
i=1
with “hole”, and each 3D Gaussian gi is defined as a se-
ries attributes gi = {µi, si, qi, shi, αi}. Then the covariance
matrix Σi ∈R3×3 of the 3D Gaussian is expressed as:
Σi = RisisT
i RT
i , where Ri ∈R3×3 is the orthogonal rotation
matrix of the Gaussian parameterized by the quaternion qi
and si ∈R3 is a scaling vector of gi. Once the Gaussian
representation is constructed, we can project the 3D Gaussian
points onto the image plane based on the given camera pose
πi ∈Π. Each Gaussian gj ∈Θ in the collection is projected
onto the image plane corresponding to the viewpoint as:
u2D
j,i = PiWiµj, Σ2D
j,i = JjWiΣT
j W T
i JT
j ,
(1)
where µ2D
j,i and Σ2D
j,i respectively represent the center and
the covariance matrix of the projected Gaussian distribution,
Wi represents the viewing transformation matrix, and Pi
represents the projective transformation matrix. Both can be
derived from the camera pose. Jj represents the Jacobian of
the affine approximation of the projective transformation. After
that, to perform image rendering on the image plane, for each
pixel p of the render image cr
i , its color cr
i (p) is derived
through an α-blending function as:
cr
i (p) =
l
X
j=1
shjβj
j−1
Y
k=1
(1 −βk),
(2)
where βj = αje−1
2 (p−µ2D
j,i )T (Σ2D
j,i )−1(x−µ2D
j,i ) and l represents
the number of projected Gaussians that overlap p. Since we
only need to reconstruct the background of the 3D Gaussian
scene, we apply a mask to ensure that the training process
focuses exclusively on the background regions while ignoring
the removed object areas. The specific loss function is formu-
lated as follows:
Linit =
N
X
i=1
∥cr
i ⊙mi −co
i ⊙mi∥2,
(3)
Fig. 3. Visualization of the Mask Refinement Process.
where ⊙is Hadamard Product. Finally, after 30,000 iterations,
we can obtain a 3D Gaussian representation with “hole”. As
shown in the middle part of Fig. 3, an example of rendering
with missing regions (“holes”) is provided.
C. Automatic Mask Refinement Process
As is well known, mainstream 3D inpainting methods rely
on multi-view 2D image inpainting. A key step is to design
an automated missing region detection algorithm for rendered
images with ”holes” in the first stage. A reliable algorithm
should retain clear background areas while accurately masking
missing regions, as shown in the final results of our method
on the right side of Fig. 3. To address the disordered floater
Gaussian kernels near the holes in the initial Gaussian (Fig. 3)
that hinder accurate “hole” detection in rendered images, we
first propose a fast filtering algorithm. Building on this, we
designed a precise automatic mask refinement module, which
will be detailed in the following part.
Gaussians Filtering. We observed that some relatively large
floating Gaussians exist in the initial Gaussian representation
Θ due to the large scope of the initial mask, as shown on the

<!-- page 5 -->
5
Fig. 4. Visualizing the effect of the Gaussian Filter: We compare the differences between the original Gaussian representation Θ and the Gaussian representation
after the Gaussian filtering operation ¯Θ in terms of rendered images, the refined masks, and the inpainted images. Here, the yellow box in the figure highlights
that our method effectively removes the floating Gaussians and achieves more accurate masks and reliable inpainted images.
left part of the Fig. 4. In other words, these floating Gaussians
occur because of insufficient multi-view training outside the
masks and the lack of guiding information within the missing
regions. These floating Gaussians can obscure background
information in certain views, thereby affecting the results of
subsequent 2D inpainting, as shown in Fig. 4. To ensure a re-
liable 2D inpainting process, we need to remove these floating
points. Thus, we assume that a valid 3D Gaussian point should
never intrude into the mask area across multiple viewpoints,
while a floating Gaussian point may appear inside the mask
in some views. Based on this assumption, we designed a fast
post-processing algorithm to remove floating kernels. . And,
this is a post-processing process that can directly accept input
from a Gaussian scene. This algorithm can directly operate on
our initial 3D Gaussian scene. Specifically, given the initial
Gaussian scene Θ, we select K key views ΠK = {πj}K
j=1
from the set of all views Π to evaluate each Gaussian kernel in
Θ. For each Gaussian point gk = {µk, sk, qk, shk, αk} ∈Θ,
its projection positions across the K key views are denoted
as µ2d
k,j. We determine whether it consistently lies outside
the corresponding masks MK = {mj}K
j=1. The filter can be
expressed as:
fmask(gk) =
K
Y
j=1
mj(µ2D
k,j),
(4)
where
mj(x) =
(
1,
if x lies outside the mask region,
0,
otherwise.
(5)
Finally, we filter out Gaussians where fmask(·) = 0 and
obtain a new Gaussian representation ¯Θ. The detailed steps
are presented in Algorithm 1.
Mask Refinement. Since our method requires sparse-view
inpainting images for 3D inpainting, a refined mask that
adequately preserves the background is essential. Predicting
masks directly from rendered images with missing parts (the
hole) typically relies on large models, which demand substan-
tial computational resources. Instead, we propose constructing
a reasonable mask directly from the previous filtered Gaus-
sian representation ¯Θ. As shown in Fig. 3, our refinement
module consists of four operations: Gaussian Projection, Local
Smoothing, Mask Intersection, and Mask Expansion.
Firstly, our Gaussian Projection can projects valid Gaussian
representation ¯Θ onto the specified viewpoints, while the non-
projected regions are highly likely to contain the real mask
regions. Specifically, given any camera position πj ∈Π , each
Gaussian point ¯gk′ ∈¯Θ can be projected onto the 2D space as
µ2D
k′,j ∈R2. Thus, we can construct a projected image mp
j ∈
RH×W as follows:
mp
j(x, y) =
(
1,
if ∃u2D
k′,j ∈[x −ϵ
2, x +
ϵ
2, ] × [y −ϵ
2, y + ϵ
2]
0,
otherwise.
(6)
Here, (x, y) represents the position of any pixel, and ϵ repre-
sents the size of a single pixel.
Secondly, a Local Smoothing operation is applied to the
discrete pixels to create continuous mask regions. We perform
convolution using 3×3 and 9×9 kernels (Conv3 and Conv9)
with all ones to compute the average value of pixels within
a local neighborhood. Here, we apply convolution operations
using smaller kernels first, followed by larger kernels, to
ensure the preservation of local mask details. The smoothed
projected image M s
j is obtained by:
ms
j = (mp
j ∗Conv3) ∗Conv9.
(7)
Subsequently, it is made to intersect with the initial mask
minter
j
. This intersection operation, denoted as minter
j
=
(ms
j ∩mj), is specifically engineered to expunge the spurious
hole regions that lie outside the purview of mj. In this step,
our negation operation mainly sets the areas inside the mask
to 0 and the areas outside the mask to 1, consistent with the
initial mask. Then, we select the largest contiguous region by
area as the mask region ¯minter
j
to remove outlier areas.
Finally, contingent upon the idiosyncratic requirements of
the specific scene under consideration, an expansion operation
may be deemed necessary for ¯minter
j
. By designating the ex-
pansion magnitude as γ, we arrive at the ultimate refined mask,
mref
j
, which is expressed as mref
j
= Expand( ¯minter
j
, γ).
Here γ is set to 15. This operation comes from a method
in the OpenCV library. The complete algorithm can be found
in the second stage of Algorithm 1.

<!-- page 6 -->
6
Algorithm 1 Automatic Mask Refinement
1: M = {mi}N
i=1 ←SAM-Track(C = {ci}N
i=1)
2: Θ ←Mask-Training(C, M)
3: ΠK = {πj}K
j=1 ←ViewSelector(Π = {πi}N
i=1)
4: Stage 1: Gaussians Filtering
5: for gk = {µk, sk, qk, ck, αk} in Θ do
6:
fmask(gk) ←1
7:
for πj in ΠK do
8:
µ2d
k,j ←proj(µk, πj)
9:
fmask(gk) = fmask(gk) · mj(µ2d
k,j)
10:
end for
11: end for
12: ¯Θ ←Remove(gk, where fmask(gk) = 1)
13: Stage 2: Mask Refinement
14: for πj in ΠK do
15:
for gk = {µk, sk, qk, ck, αk} in ¯Θ do
16:
µ2d
k,j ←proj(µk, πj)
17:
(x, y) ←Convert µ2d
k,j to pixel coordinates
18:
mp
j ←Calculate using Eq. 6
19:
end for
20:
ms
j ←Calculate using Eq. 7
21:
minter
j
←Intersection(ms
j,mj)
22:
mref
j ←Expand(minter
j
, γ)
23: end for
Fig. 5.
Visualization of Uncertainty Optimization Results Initialized from
Depth. We compare results with and without the regularization term. The red
boxes highlight that without the regularization term, dense uncertainty regions
lead to more chaotic Gaussian field estimation.
D. Uncertainty-Based Sparse View Consistency Inpainting
Diffusion-based Depth and Image Inpainting. After remov-
ing the cluttered Gaussian points from the Gaussian scene
and obtaining refined masks and rendered RGBD images
CK = {cj}K
j=1 and MK = {mj}K
j=1 for several keyframes,
we proceed with the steps outlined in Infusion [19], utilizing a
diffusion model and its depth completion model to diffuse the
RGBD data of these keyframes. This process yields richly de-
tailed RGB images and smooth, completed depth maps Cin
K =
{cin
j }K
j=1 and Din
K = {din
j }K
j=1, respectively. The challenge
lies in leveraging these inconsistent 2D results to supervise
the training of the Gaussian scene for a consistent 3D scene.
Inspired by [20], we introduce a mechanism based on the
pixel-level uncertainty of primary and secondary viewpoints
to harmoniously integrate these inconsistent images, ultimately
achieving a complete 3D Gaussian scene Θinpainted, thereby
overcoming this challenge.
Uncertainty-guided Fine-grained Optimization. As shown
in Fig. 2, following the process outlined in Infusion [19],
we first back-project the RGB image from the primary view
into the Gaussian scene using the inpained depth and images
{cin
j }K
j=1 and {din
j }K
j=1. However, for large-scale missing
regions, relying solely on a single view information for re-
construction is insufficient to ensure reliability across multiple
views. Furthermore, the introduction of multiple viewpoints
increases inconsistency, as shown in Fig. 1. To address this
challenge, we propose to leverage multi-view uncertainty to
assign unfilled regions in the primary view to other views
for 3D scene inpainting. Specifically, areas with lower depth
values in the primary view are generally associated with
higher confidence and clearer details, making them more
reliable for reconstruction. Conversely, regions with larger
depth values are more difficult to complete and can benefit
from complementary information provided by other views. To
ensure consistency across regions, we introduce an uncertainty
mechanism, initializing the uncertainty within the refined
masks on the inpainted depths Din
K = {din
j }K
j=1 from key
views.
For optimizing each key viewpoint, we adopted a fine-
grained optimization strategy guided by uncertainty. Specif-
ically, given the key viewpoint ΠK, the refined masks M ref
K
computed in the previous steps, the inpainted images Cin
K , and
the predicted depths Din
K generated by the diffusion model, we
proceed as follows: For the j-th view πj ∈ΠK,we define fine-
grained uncertainty values with resolution r for a key image
cin
j
∈Cin
K , we represent Uj ∈R
H
r × W
r , and the uncertainty
values are first initialized using the predicted depth din
j ∈Din
K ,
as expressed by:
Uj[hr, wr] = λ · mean(din
j [hr × 8 : (hr + 1) × 8,
wr × 8 : (wr + 1) × 8]),
(8)
where λ controls the initialization scale to ensure the opti-
mization process converges within a suitable range, balancing
convergence speed and model stability. Here, we perform
block-based optimization of the uncertainty values to improve
training stability. Point-wise optimization can lead to instabil-
ity in model optimization.
The confidence weights Wj ∈RH×W are then defined as:
Wj[hr × 8 : (hr + 1) × 8,
wr × 8 : (wr + 1) × 8] =
1
Uj[hr, wr],
(9)
where hr ∈[0, H
r −1] and nr ∈[0, W
r −1].

<!-- page 7 -->
7
The overall loss function is expressed as:
Luncertainty =
X
πj∈ΠK
[
mref
j ⊙W2
j
2
⊙(cr
j −cin
j )

2
2
+
X
mref
j [h,w]̸=0
log

1
Wj[h, w]

],
(10)
where M ref
j
denotes the refined mask and cr
j is denoted the
rendered image. The optimization focuses exclusively on the
uncertainties within the regions defined by the mask. It is
worth noting that the second term acts as a regularizer, promot-
ing sparsity in the uncertainty distribution. Ideally, uncertainty
should be concentrated in regions far from the viewpoint. A
dense uncertainty map would indicate chaotic or unreliable
observations, undermining the model’s effectiveness.
With the introduction of the uncertainty loss, our overall
loss function is formulated as:
L = λ1Lrec + λ2Ldepth + λ3Luncertainty,
(11)
where the coefficients λ1, λ2, and λ3 control the relative con-
tribution of each term. In our experiments, they are empirically
set to 1, 0.5, and 1, respectively.
Among them, the reconstruction loss Lrec consists of two
components: the reconstruction loss for the background region
and the reconstruction loss for the main reference view, which
represents the most representative frame among the selected
key views. It is defined as:
Lrec = Lbg
rec + Lref
rec .
(12)
Specifically, the reconstruction constraint for the back-
ground region is defined as:
Lbg
rec =
X
πj∈Π/ΠK
h
∥¯mr
j ⊙(cr
j −c
¯Θ
j )∥1
+ D-SSIM( ¯mr
j ⊙cr
j, ¯mr
j ⊙c
¯Θ
j )
i
.
(13)
Here, ¯mr
j denotes the background region in the refined mask,
and c ¯Θ
j represents the image rendered from the filtered Gaus-
sians, serving as the supervisory signal.
The reconstruction loss for the main reference view is given
by:
Lref
rec =∥cr
ref −cin
ref∥1
+ D-SSIM(cr
ref, cin
ref)
+ λ4LPIPS(cr
ref, cin
ref).
(14)
Here, cref denotes the primary view among the selected
keyframes and λ4 is set to 0.5. An additional constraint is
applied to this view to enhance reconstruction quality from
this critical perspective.
To enforce geometric consistency, we also introduce a depth
supervision loss for keyframes:
Ldepth =
X
πj∈ΠK
∥dr
j −din
j ∥1,
(15)
where dr
j is the depth map rendered from the Gaussians and
din
j
denotes the generated pseudo ground-truth depth. Finally,
the uncertainty is updated using a gradient descent algorithm,
specifically leveraging the Adam optimizer with an initial
learning rate of 0.02. Throughout training, inconsistent regions
within the key views are progressively refined, leading to
updated uncertainty estimates. This dynamic adjustment en-
courages the model to focus on consistent regions, ultimately
balancing multi-view consistency with the preservation of fine-
grained details.
Based on the above loss functions, we iteratively update the
uncertainty values, as illustrated in Fig. 5, which visualizes
the final uncertainty distributions across multiple key views.
The experimental results indicate that a sparse uncertainty
prediction effectively reduces multi-view inconsistency and
mitigates the resulting disorder in Gaussian field estimation.
IV. EXPERIMENT
A. Datasets and Settings
To evaluate the effectiveness of our proposed algorithm,
we conduct experiments on three representative datasets.
Specifically, SPIn-NeRF [21] comprises 10 front-facing wide-
field scenes, including 7 outdoor and 3 indoor scenes. Each
scene consists of 60 training images and 40 test images,
accompanied by binary masks and ground-truth images for
object removal evaluation. In addition, we utilize the “kitchen”
scene from Mip-NeRF360 [56] and the “bear” and “garden”
scenes from InNeRF360 [57]. These datasets are characterized
by large view-angle displacements, covering 360° of camera
poses, making them suitable for evaluating the robustness of
object removal in challenging scenarios. Due to the lack of
ground-truth object-removed images in these two datasets,
our evaluation mainly relies on qualitative comparisons to
demonstrate the effectiveness of our method under substantial
viewpoint changes. Since ground-truth masks are also unavail-
able in these datasets, we employ SAM-Track [58] to generate
initial rough masks of the target objects for each frame as input
for rendering process.
All experiments are conducted on a single RTX 3090 GPU
with 24GB VRAM. The initial optimization is performed
for 30,000 iterations with a learning rate of 0.02. For the
SPIn-NeRF dataset [21], 4–6 reference views are used during
the mask refinement stage, followed by 1,500 iterations of
second-stage optimization using 2 sparse views. For larger-
scale scenes in other datasets [56, 57], approximately 10 views
are used for refinement, and 4 sparse views are employed for
the second stage, which runs for 10,000 iterations.
B. Quantitative Evaluations
As shown in Tab. I, we present a quantitative comparison
of our method against several related approaches. These in-
clude NeRF-based methods such as SPIn-NeRF [21] and OR-
NeRF [49], as well as Gaussian Splatting-based approaches
like Gaussian Grouping [24], GScream [54], and Infusion [19].
Among these methods, only GScream and Infusion utilize
single-view Stable Diffusion (SD) for inpainting, thereby
avoiding the challenge of multi-view inconsistency. However,
this also leads to degraded synthesis quality in distant or novel
views, due to the lack of multi-view contextual information
and geometric consistency. In contrast, the other methods rely

<!-- page 8 -->
8
Fig. 6. Qualitative Comparison of Object Removal and Inpainting Methods. Illustration of rendered images with object removal and inpainting, compared
with SPIn-NeRF [21], OR-NeRF [49], and Gaussian splatting-based methods, including Infusion [19], GScream [54], and Gaussian Group [24]. Among these
methods, all except Infusion [19] and GScream [54] use LaMa [17] for image inpainting, while Infusion [19] and GScream [54] rely on single-view Stable
Diffusion [55] for inpainting. The results highlight the effectiveness of our approach in achieving a natural and seamless object removal effect.
on LaMa [17] to inpaint multiple views, which may introduce
noticeable blurriness in the reconstructed scene.
To evaluate the quality of novel view synthesis, we follow
previous works and adopt the LPIPS (Learned Perceptual
Image Patch Similarity) and FID (Frechet Inception Distance)
metrics. Specifically, LPIPS leverages pre-trained deep neural
networks to extract image features and computes the percep-
tual similarity by measuring distances in the feature space,
closely aligning with human visual perception. FID, on the
other hand, quantifies the statistical difference between feature
distributions of real and generated images using a pre-trained
Inception network, where lower scores indicate higher visual
fidelity.
For a fair comparison, we also integrate LaMa [17] into
our pipeline to perform inpainting on sparse views. The
experimental results demonstrate that our method achieves
competitive performance across most metrics. Although our
approach generally outperforms other baselines, the FID score
is slightly higher than that of GScream [54]. This can be
attributed to GScream’s use of anchor-based constraints, which
help maintain consistency in the Gaussian scene representation
and mitigate deviations from the ground-truth distribution.
Additionally, our method exhibits clear advantages in compu-
tational efficiency, highlighting its practicality for real-world
applications.
C. Qualitative Results
For qualitative evaluation, we compare our method with rep-
resentative baselines on the SPIn-NeRF dataset. In our setup,
two LaMa [17] inpainted views are used for the second-stage
optimization. In contrast, most competing methods (except
Infusion and GScream) rely on more inpainted views, which
may introduce redundancy and multi-view inconsistencies.
SPIn-NeRF [21] and OR-NeRF [49], both NeRF-based
methods, suffer from blurry renderings due to limited spatial
resolution and weak multi-view consistency. SPIn-NeRF lacks
effective cross-view constraints, often leading to appearance
artifacts. OR-NeRF removes foregrounds but fails to preserve
fine details, with LaMa inpainting yielding overly smoothed
results and degraded scene fidelity (see Fig. 6).
Infusion [19], relying on single-view depth estimation and
back-projection, struggles at mask boundaries, resulting in
unrealistic edges and strong artifacts. GScream [54], also
guided by a single view, often exhibits tearing and ghosting.
While it uses anchor point constraints to maintain semantics,
these can cause structural duplications and unnatural object
extensions (highlighted in red boxes in Fig. 6).
Lastly, Gaussian Grouping [24] performs poorly on SPIn-
NeRF due to inaccurate object segmentation. The resulting
incomplete masks impair inpainting quality, leading to failed
reconstructions or heavily blurred outputs.
Furthermore, we leverage a diffusion-based inpainting
model [55], to repair occluded regions in the input images.

<!-- page 9 -->
9
TABLE I
QUANTITATIVE RESULTS OF NOVEL VIEW SYNTHESIS AFTER OBJECT
REMOVAL. WE CONDUCT A COMPARATIVE STUDY INVOLVING
NERF-BASED METHODS (SPIN-NERF [21] AND OR-NERF [49]) AND
GAUSSIAN SPLATTING-BASED APPROACHES (INFUSION [19],
GSCREAM [54], AND GAUSSIAN GROUP [24]). IN THIS STUDY, WE
UTILIZE LAMA [17] FOR IMAGE INPAINTING TO HANDLE MISSING
REGIONS.
Method
LPIPS↓
FID↓
Time↓
SPIn-NeRF [21]
0.47
147.31
4h
OR-NeRF [49]
0.56
38.69
4h
Gaussian Grouping [24]
0.45
123.48
20min
GScream [54]
0.28
36.72
1.2h
Infusion [19]
0.42
92.62
40s
Ours
0.22
55.17
3min
Fig. 7. Scene Completion Results Using Stable Diffusion-Based Inpainting.
For each example, we present the reconstructed scene from two different
viewpoints to illustrate the effectiveness of image inpainting in filling missing
regions across multiple perspectives. Our method produces consistent and
more faithful scene completions, particularly in complex scenarios.
We conduct visual comparisons across several complex scenes
to evaluate the effectiveness of our approach. As illustrated
in Fig. 7, our method achieves more realistic scene recon-
structions with richer texture details. Benefiting from the
proposed uncertainty-guided constraint mechanism and more
accurate mask estimation, our approach generates multi-view-
consistent results that surpass those of Infusion, which relies
solely on single-view depth estimation. Notably, our method
is able to maintain both high visual clarity and detailed texture
continuity across views, demonstrating superior generalization
in challenging object removal scenarios.
TABLE II
ABLATION STUDY OF UNCERTAINTY-GUIDED FINE-GRAINED
OPTIMIZATION. COMPARISON WITH NON-UNCERTAINTY-GUIDED AND
NON-DEPTH INITIALIZED STRATEGIES IN TWO SCENES.
Scenes
LPIPS↓
FID↓
Scene A
w/o uncertainty
0.22
35.58
w/o depth Init.
0.21
31.29
w/ depth + uncertainty
0.17
31.72
Scene B
w/o uncertainty
0.21
32.63
w/o depth init
0.21
30.28
w/ depth + uncertainty
0.18
28.04
Fig. 8. Effectiveness of Mask Refinement on Image Inpainting. Qualitative
ablation study demonstrating the effectiveness of our mask refinement process
for image inpainting. The first column shows the original frames. The second
column presents inpainting results using only coarse masks, which often lead
to incomplete or unrealistic completions. The third column shows the results
with our refined masks, which enable more accurate and visually consistent
image restoration, especially in complex scenes.
D. Ablation Study
To further validate the effectiveness of our proposed
uncertainty-guided optimization strategy, we conducted both
quantitative and qualitative ablation studies. Specifically, for
the first and last scenes illustrated in Fig. 6, we compare our
method against two baseline variants: one without uncertainty
guidance and another without using depth estimation for
initialization. As shown in Tab. II, our method consistently
outperforms both baselines, demonstrating the effectiveness
of each design component. In addition, the second row in
Fig. 9 provides a qualitative comparison that highlights the role
of the uncertainty mechanism. Without uncertainty guidance,
training the Gaussian scene using multiple inpainted images
leads to inconsistencies in intermediate views. This results in
visual artifacts and incoherent textures, particularly in regions
farther from the primary views. In contrast, our uncertainty-
aware approach enables the network to prioritize supervision
from views with high confidence, ensuring that textures from
the primary view dominate in shared regions, while comple-

<!-- page 10 -->
10
Fig. 9.
Visual Ablation Study on the Effectiveness of Refined Mask and
Uncertainty-Guided Optimization in Gaussian Scene Refinement. The first
row shows the optimization results without our refined mask strategy, leading
to inaccurate and incomplete updates. The second row removes the uncertainty
guidance, resulting in suboptimal consistency. The third row displays our full
method, demonstrating improved precision and coherence in Gaussian scene
reconstruction.
Fig. 10. Comparison of rendered depth and RGB images from two viewpoints,
with and without the proposed uncertainty constraint applied to depth supervi-
sion. Results demonstrate that introducing uncertainty into depth supervision
destabilizes the optimization process.
mentary views contribute only to filling occluded or missing
areas. This strategy helps maintain consistency and realism in
the synthesized scene.
To assess the effectiveness of the mask refinement process,
we conduct experiments on the “bear” scene. As shown in
Fig. 8, utilizing our refined masks during the inpainting stage
significantly improves the quality and consistency of the re-
stored content. When using coarse masks directly, the diffusion
model tends to overextend into occluded regions behind the
removed objects, introducing semantic inconsistencies and
hallucinated content. As shown in Fig. 8, the region behind the
sculpture is incorrectly inpainted with semantically unrelated
elements, leading to structural conflicts in the final reconstruc-
tion. Furthermore, as illustrated in the first row of Fig. 9,
unrefined masks result in inaccurate inpainting, which directly
impacts the gaussian scene training by introducing conflicting
information between the restored and original regions. This
leads to blurred or inconsistent reconstructions. Our mask
refinement strategy, on the other hand, better isolates the target
object, reduces unnecessary modification of the scene, and thus
minimizes artifacts caused by inconsistent supervision.
It is worth noting that the estimated uncertainty is applied
only to the RGB images between key views to ensure tex-
ture consistency. Depth supervision primarily provides coarse
geometric guidance, and minor inconsistencies across views
have limited impact on the final reconstruction. In contrast,
introducing uncertainty constraints into the depth supervision
leads to unstable optimization. As shown in Fig. 10, the
rendered depth maps and final Gaussian scenes optimized with
uncertainty-guided depth supervision exhibit degraded perfor-
mance, suggesting that such constraints hinder convergence.
To assess the impact of key view selection on inpainting
quality, we conducted visual comparisons using 2, 4, and 8
spatially distributed viewpoints, as illustrated in Fig. 11. The
results show that using too few views leads to insufficient
scene coverage, while using too many can introduce geometric
inconsistencies, especially in complex 360-degree environ-
ments. Based on these observations, we select approximately
10% of all available views (typically 4 views) as key view-
points for large-scale scenes, and 2 views for smaller scenes
such as those in the SPIn-NeRF dataset. Additionally, about
30% of the views are employed during the mask refinement
stage to ensure adequate spatial guidance.
V. LIMITATIONS AND FUTURE WORK
Although the proposed method proves effective, it still has
certain limitations. One major challenge is the presence of
large and uncontrollable appearance variations among views.
To address this issue, future work may explore video diffu-
sion models guided by auxiliary modalities such as normal
maps, depth maps, semantic labels, and texture cues to better
preserve geometric and semantic consistency across views.
Incorporating a cross-view attention mechanism to diffuse
multiple views jointly under consistency constraints, together
with Score Distillation Sampling (SDS) loss for auxiliary
supervision, could further enhance inter-view coherence. Ad-
ditionally, extending the framework to 3D object replacement
is a promising direction, where joint attention and multi-modal
reasoning may ensure consistent and realistic inpainting views
across different viewpoints.
VI. CONCLUSION
We propose a sparse image-guided 3D Gaussian Inpainting
framework. Specifically, we introduce an automatic Mask
Refinement module for estimating regions to be inpainted in
the initial scene, and to reduce the influence of noisy Gaussian
points on the mask estimation, a Gaussian Filtering opera-
tion is also applied. This mask refinement technique ensures
more accurate mask estimation, significantly improving the
boundary quality of the inpainted regions in the scene. Addi-
tionally, to address multi-view inconsistencies, we present an
Uncertainty-guided Fine-grained Optimization method. This
technique automatically estimates the contribution of each
pixel to the scene optimization during the Gaussian render-
ing update process, mitigating conflicts between multi-view
images. In our experiments, we demonstrate both quantita-
tively and qualitatively that our framework can handle scenes
from various camera viewpoints and outperforms existing 3D
inpainting methods.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron,
R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes
as neural radiance fields for view synthesis,” Communi-
cations of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

<!-- page 11 -->
11
Fig. 11. Comparison of inpainting results using 2, 4, and 8 key views selected from eight candidate views. The results demonstrate that an appropriate number
of key views is critical for inpainting quality, as too few result in incomplete coverage while too many can introduce optimization ambiguity in complex
scenes.
[2] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman,
R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A
multiscale representation for anti-aliasing neural radiance
fields,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 5855–5864.
[3] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and
W. Wang, “Neus: Learning neural implicit surfaces by
volume rendering for multi-view reconstruction,” arXiv
preprint arXiv:2106.10689, 2021.
[4] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis,
“3d gaussian splatting for real-time radiance field ren-
dering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1,
2023.
[5] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d
gaussian splatting for geometrically accurate radiance
fields,” in ACM SIGGRAPH 2024 Conference Papers,
2024, pp. 1–11.
[6] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaus-
sian splatting for efficient 3d mesh reconstruction and
high-quality mesh rendering,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024, pp. 5354–5363.
[7] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and
B. Dai, “Scaffold-gs: Structured 3d gaussians for view-
adaptive rendering,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2024, pp. 20 654–20 664.
[8] J. Chung, S. Lee, H. Nam, J. Lee, and K. M. Lee, “Lucid-
dreamer: Domain-free generation of 3d gaussian splatting
scenes,” arXiv preprint arXiv:2311.13384, 2023.
[9] T. Yi, J. Fang, J. Wang, G. Wu, L. Xie, X. Zhang,
W. Liu, Q. Tian, and X. Wang, “Gaussiandreamer: Fast
generation from text to 3d gaussians by bridging 2d and
3d diffusion models,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2024, pp. 6796–6807.
[10] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu,
Q. Tian, and X. Wang, “4d gaussian splatting for real-
time dynamic scene rendering,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024, pp. 20 310–20 320.
[11] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-
time photorealistic dynamic scene representation and
rendering with 4d gaussian splatting,” arXiv preprint
arXiv:2310.10642, 2023.
[12] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow:
4d reconstruction with dynamic 3d gaussian particle,” in
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 21 136–21 145.
[13] T. Zhang, H.-X. Yu, R. Wu, B. Y. Feng, C. Zheng,
N. Snavely, J. Wu, and W. T. Freeman, “Physdreamer:
Physics-based interaction with 3d objects via video gen-
eration,” in European Conference on Computer Vision.
Springer, 2025, pp. 388–406.
[14] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang,
and C. Jiang, “Physgaussian: Physics-integrated 3d gaus-
sians for generative dynamics,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024, pp. 4389–4398.
[15] Y. Jiang, C. Yu, T. Xie, X. Li, Y. Feng, H. Wang, M. Li,
H. Lau, F. Gao, Y. Yang et al., “Vr-gs: A physical
dynamics-aware interactive gaussian splatting system in
virtual reality,” in ACM SIGGRAPH 2024 Conference
Papers, 2024, pp. 1–1.
[16] J. Jam, C. Kendrick, K. Walker, V. Drouard, J. G.-S.
Hsu, and M. H. Yap, “A comprehensive review of past
and present image inpainting methods,” Computer vision
and image understanding, vol. 203, p. 103147, 2021.
[17] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova,
A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park,
and V. Lempitsky, “Resolution-robust large mask inpaint-
ing with fourier convolutions,” in Proceedings of the
IEEE/CVF winter conference on applications of com-
puter vision, 2022, pp. 2149–2159.
[18] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and
B. Ommer, “High-resolution image synthesis with la-
tent diffusion models,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,

<!-- page 12 -->
12
2022, pp. 10 684–10 695.
[19] Z. Liu, H. Ouyang, Q. Wang, K. L. Cheng, J. Xiao,
K. Zhu, N. Xue, Y. Liu, Y. Shen, and Y. Cao, “Infusion:
Inpainting 3d gaussians via learning depth completion
from diffusion prior,” arXiv preprint arXiv:2404.11613,
2024.
[20] S. Weder, G. Garcia-Hernando, A. Monszpart, M. Polle-
feys, G. J. Brostow, M. Firman, and S. Vicente, “Remov-
ing objects from neural radiance fields,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 16 528–16 538.
[21] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpa-
nis, J. Kelly, M. A. Brubaker, I. Gilitschenski, and
A. Levinshtein, “Spin-nerf: Multiview segmentation and
perceptual inpainting with neural radiance fields,” in
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 20 669–20 679.
[22] J. Wang, J. Fang, X. Zhang, L. Xie, and Q. Tian,
“Gaussianeditor: Editing 3d gaussians delicately with text
instructions,” in Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2024,
pp. 20 902–20 911.
[23] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang,
Z. Cai, L. Yang, H. Liu, and G. Lin, “Gaussianeditor:
Swift and controllable 3d editing with gaussian splat-
ting,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp.
21 476–21 485.
[24] M. Ye, M. Danelljan, F. Yu, and L. Ke, “Gaussian
grouping: Segment and edit anything in 3d scenes,” in
European Conference on Computer Vision.
Springer,
2025, pp. 162–179.
[25] J. Huang, H. Yu, J. Zhang, and H. Nait-Charif, “Point’n
move: Interactive scene object manipulation on gaussian
splatting radiance fields,” IET Image Processing, 2024.
[26] W. Quan, J. Chen, Y. Liu, D.-M. Yan, and P. Wonka,
“Deep learning-based image and video inpainting: A
survey,” International Journal of Computer Vision, vol.
132, no. 7, pp. 2367–2400, 2024.
[27] O. Elharrouss, N. Almaadeed, S. Al-Maadeed, and Y. Ak-
bari, “Image inpainting: A review,” Neural Processing
Letters, vol. 51, pp. 2007–2028, 2020.
[28] C. Ballester, M. Bertalmio, V. Caselles, G. Sapiro, and
J. Verdera, “Filling-in by joint interpolation of vector
fields and gray levels,” IEEE transactions on image
processing, vol. 10, no. 8, pp. 1200–1211, 2001.
[29] D. Tschumperl´e and R. Deriche, “Vector-valued image
regularization with pdes: A common framework for
different applications,” IEEE transactions on pattern
analysis and machine intelligence, vol. 27, no. 4, pp.
506–517, 2005.
[30] A. A. Efros and T. K. Leung, “Texture synthesis by
non-parametric sampling,” in Proceedings of the seventh
IEEE international conference on computer vision, vol. 2.
IEEE, 1999, pp. 1033–1038.
[31] C. Barnes, E. Shechtman, A. Finkelstein, and D. B.
Goldman, “Patchmatch: a randomized correspondence
algorithm for structural image editing,” ACM Trans.
Graph., vol. 28, no. 3, 2009.
[32] S. Darabi, E. Shechtman, C. Barnes, D. B. Goldman,
and P. Sen, “Image melding: Combining inconsistent
images using patch-based synthesis,” ACM Transactions
on graphics (TOG), vol. 31, no. 4, pp. 1–10, 2012.
[33] J.-B. Huang, S. B. Kang, N. Ahuja, and J. Kopf, “Im-
age completion using planar structure guidance,” ACM
Transactions on graphics (TOG), vol. 33, no. 4, pp. 1–
10, 2014.
[34] J. Herling and W. Broll, “High-quality real-time video
inpaintingwith pixmix,” IEEE Transactions on Visualiza-
tion and Computer Graphics, vol. 20, no. 6, pp. 866–879,
2014.
[35] Q. Guo, S. Gao, X. Zhang, Y. Yin, and C. Zhang,
“Patch-based image inpainting via two-stage low rank
approximation,” IEEE transactions on visualization and
computer graphics, vol. 24, no. 6, pp. 2023–2036, 2017.
[36] Y. Wexler, E. Shechtman, and M. Irani, “Space-time
completion of video,” IEEE Transactions on pattern
analysis and machine intelligence, vol. 29, no. 3, pp.
463–476, 2007.
[37] M. Granados, K. I. Kim, J. Tompkin, J. Kautz, and
C. Theobalt, “Background inpainting for videos with
dynamic objects and a free-moving camera,” in Com-
puter Vision–ECCV 2012: 12th European Conference on
Computer Vision, Florence, Italy, October 7-13, 2012,
Proceedings, Part I 12.
Springer, 2012, pp. 682–695.
[38] A. Newson, A. Almansa, M. Fradet, Y. Gousseau, and
P. P´erez, “Video inpainting of complex scenes,” Siam
journal on imaging sciences, vol. 7, no. 4, pp. 1993–
2019, 2014.
[39] J.-B. Huang, S. B. Kang, N. Ahuja, and J. Kopf, “Tem-
porally coherent completion of dynamic video,” ACM
Transactions on Graphics (ToG), vol. 35, no. 6, pp. 1–11,
2016.
[40] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt,
B. Dai, and D. Lin, “Bungeenerf: Progressive neural
radiance field for extreme multi-scale scene rendering,”
in European conference on computer vision.
Springer,
2022, pp. 106–122.
[41] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht,
and A. Kanazawa, “K-planes: Explicit radiance fields
in space, time, and appearance,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 12 479–12 488.
[42] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant
neural graphics primitives with a multiresolution hash en-
coding,” ACM transactions on graphics (TOG), vol. 41,
no. 4, pp. 1–15, 2022.
[43] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d
gaussian splatting for geometrically accurate radiance
fields,” in SIGGRAPH 2024 Conference Papers.
As-
sociation for Computing Machinery, 2024.
[44] A. Gu´edon and V. Lepetit, “Gaussian frosting: Editable
complex radiance fields with real-time rendering,” in
European Conference on Computer Vision.
Springer,
2025, pp. 413–430.
[45] T. Xu, J. Chen, P. Chen, Y. Zhang, J. Yu, and W. Yang,

<!-- page 13 -->
13
“Tiger: Text-instructed 3d gaussian retrieval and coherent
editing,” arXiv preprint arXiv:2405.14455, 2024.
[46] Q. Zhang, Y. Xu, C. Wang, H.-Y. Lee, G. Wetzstein,
B. Zhou, and C. Yang, “3ditscene: Editing any scene via
language-guided disentangled gaussian splatting,” arXiv
preprint arXiv:2405.18424, 2024.
[47] H.-K. Liu, I. Shen, B.-Y. Chen et al., “Nerf-in: Free-
form nerf inpainting with rgb-d priors,” arXiv preprint
arXiv:2206.04901, 2022.
[48] A. Mirzaei, T. Aumentado-Armstrong, M. A. Brubaker,
J.
Kelly,
A.
Levinshtein,
K.
G.
Derpanis,
and
I. Gilitschenski, “Reference-guided controllable inpaint-
ing of neural radiance fields,” in Proceedings of the
IEEE/CVF International Conference on Computer Vi-
sion, 2023, pp. 17 815–17 825.
[49] Y. Yin, Z. Fu, F. Yang, and G. Lin, “Or-nerf: Object
removing from 3d scenes guided by multiview seg-
mentation with neural radiance fields,” arXiv preprint
arXiv:2305.10503, 2023.
[50] Y. Lu, J. Ma, and Y. Yin, “View-consistent object re-
moval in radiance fields,” in Proceedings of the 32nd
ACM International Conference on Multimedia, 2024, pp.
3597–3606.
[51] Y. Wang, Q. Wu, G. Zhang, and D. Xu, “Learning 3d
geometry and feature consistent gaussian splatting for
object removal,” in European Conference on Computer
Vision.
Springer, 2025, pp. 1–17.
[52] J. Huang, H. Yu, J. Zhang, and H. Nait-Charif, “Point’n
move: Interactive scene object manipulation on gaussian
splatting radiance fields,” IET Image Processing, 2023.
[53] S.-Y. Huang, Z.-T. Chou, and Y.-C. F. Wang, “3d gaus-
sian inpainting with depth-guided cross-view consis-
tency,” in Proceedings of the Computer Vision and Pat-
tern Recognition Conference, 2025, pp. 26 704–26 713.
[54] Y. Wang, Q. Wu, G. Zhang, and D. Xu, “Gscream:
Learning 3d geometry and feature consistent gaussian
splatting for object removal,” in ECCV, 2024.
[55] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and
B. Ommer, “High-resolution image synthesis with latent
diffusion models,” 2021.
[56] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan,
and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased
neural radiance fields,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
2022, pp. 5470–5479.
[57] D. Wang, T. Zhang, A. Abboud, and S. S¨usstrunk,
“Innerf360: Text-guided 3d-consistent object inpainting
on 360-degree neural radiance fields,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 12 677–12 686.
[58] Y. Cheng, L. Li, Y. Xu, X. Li, Z. Yang, W. Wang, and
Y. Yang, “Segment and track anything,” arXiv preprint
arXiv:2305.06558, 2023.
