<!-- page 1 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
Yuxin Cheng
yxcheng@connect.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Binxiao Huang
huangbx7@connect.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Wenyong Zhou
wenyongz@connect.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Taiqiang Wu
takiwu@connect.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Zhengwu Liu
zwliu@eee.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Graziano Chesi
chesi@eee.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Ngai Wong∗
nwong@eee.hku.hk
The University of Hong Kong
Hong Kong SAR, China
Renderings
(a) SFM initialization and Ground Truth
GT
(b) 3D-GS
PSNR: 21.46
SSIM: 0.611
LPIPS: 0.332
(c) ReAct-GS w/o re-activation (ours)
PSNR: 21.74
SSIM: 0.644
LPIPS: 0.253
(d) ReAct-GS full (ours)
PSNR: 21.91
SSIM: 0.658
LPIPS: 0.240
Pipeline
initial 3D Gaussian 
primitives
average gradient 
densification
clone
small-scale
large-scale
split
importance-aware
densification
clone
small-scale
large-scale
split
exhibit freezing locally
re-activate
fit better globally
low 
density
needle 
shape
3D-GS
ReAct-GS (ours)
additional 
important 
primitives
adaptive re-activation
selected
to densify
Figure 1: We present ReAct-GS, a method that addresses the over-reconstruction issue in 3D-GS by identifying and resolving two
fundamental limitations: gradient magnitude dilution and primitive frozen phenomenon. We introduce a novel importance-
aware densification criterion and an adaptive re-activation mechanism that effectively eliminate over-reconstruction artifacts
in complex scenes, achieving improved rendering quality while accurately preserving fine details in high-frequency regions.
The code will be publicly available at https://react-gs.github.io.
Abstract
3D Gaussian Splatting (3D-GS) achieves real-time photorealistic
novel view synthesis, yet struggles with complex scenes due to
over-reconstruction artifacts, manifesting as local blurring and
needle-shape distortions. While recent approaches attribute these
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
MM ’25, Dublin, Ireland
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2035-2/2025/10
https://doi.org/10.1145/3746027.3754958
issues to insufficient splitting of large-scale Gaussians, we identify
two fundamental limitations: gradient magnitude dilution during
densification and the primitive frozen phenomenon, where essen-
tial Gaussian densification is inhibited in complex regions while
suboptimally scaled Gaussians become trapped in local optima. To
address these challenges, we introduce ReAct-GS, a method founded
on the principle of re-activation. Our approach features: (1) an
importance-aware densification criterion incorporating 𝛼-blending
weights from multiple viewpoints to re-activate stalled primitive
growth in complex regions, and (2) a re-activation mechanism
that revitalizes frozen primitives through adaptive parameter per-
turbations. Comprehensive experiments across diverse real-world
datasets demonstrate that ReAct-GS effectively eliminates over-
reconstruction artifacts and achieves state-of-the-art performance
on standard novel view synthesis metrics while preserving intri-
cate geometric details. Additionally, our re-activation mechanism
arXiv:2510.19653v1  [cs.CV]  21 Oct 2025

<!-- page 2 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
yields consistent improvements when integrated with other 3D-GS
variants such as Pixel-GS, demonstrating its broad applicability.
CCS Concepts
• Computing methodologies →Rendering; Machine learning
approaches; Point-based models.
Keywords
Novel View Synthesis, 3D Reconstruction, 3D Gaussian Splatting
ACM Reference Format:
Yuxin Cheng, Binxiao Huang, Wenyong Zhou, Taiqiang Wu, Zhengwu Liu,
Graziano Chesi, and Ngai Wong. 2025. Re-Activating Frozen Primitives
for 3D Gaussian Splatting. In Proceedings of the 33rd ACM International
Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland.
ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3746027.3754958
1
Introduction
The advancements in 3D Gaussian Splatting (3D-GS) [10] have
established it as a promising approach for explicit point-based
neural representation, demonstrating significant potential across
applications from immersive VR/AR to robotics [15, 26, 33]. This
representation paradigm enables exceptionally rapid training and
real-time rendering while maintaining photorealistic quality. These
capabilities stem from 3D-GS’s distinctive ability to adaptively
refine Gaussian primitives through densification and optimization,
offering an optimal balance between computational efficiency and
visual fidelity. However, when applied to complex scenarios (e.g.,
unbounded outdoor or high-frequency texture), 3D-GS exhibits
persistent over-reconstruction artifacts. These artifacts manifest as
local blurring and loss of fine details, primarily attributed to regions
dominated by suboptimally scaled Gaussians (either oversized or
needle-shape). Addressing these quality degradation issues has
become a critical challenge in advancing 3D-GS technology.
Recent research has primarily focused on aggressive splitting
strategies to mitigate over-reconstruction in 3D-GS. Notable ap-
proaches such as Abs-GS [29] and Mini-Splatting [5] propose large-
scale Gaussian splitting to reduce rendering artifacts caused by
oversized primitives. While these methods achieve improved visual
quality, they suffer from a critical limitation: excessive splitting
leads to premature fragmentation of background primitives and
potentially biases the growth of 3D Gaussians toward physically
implausible locations as an unintended consequence (see Figure 2).
In this study, we identify two fundamental limitations in the
current 3D-GS pipeline that collectively cause over-reconstruction.
Through systematic analysis, we first demonstrate that the original
average gradient criterion suffers from gradient magnitude dilu-
tion. We elucidate that gradient magnitude strongly correlates with
a 3D Gaussian’s contribution to 𝛼-blending from corresponding
viewpoints. This correlation causes critical growth signals from
dominant viewpoints to be suppressed when averaged with weaker
gradients from less influential perspectives, thereby inhibiting den-
sification in regions requiring fine detail representation. More criti-
cally, we discover the previously overlooked primitive frozen phe-
nomenon, where rapidly converged small-scale and needle-shape
3D Gaussians frequently become trapped in optimization stagna-
tion. Their contracted receptive fields and attenuated gradients
Large-scale split
ReAct-GS (Ours)
(a) Renderings
(b) Depth maps
(c) Point clouds below the horizon
Figure 2: Visualization of renderings, depth maps and point
cloud distributions in an unbounded 3D scene. The method
Abs-GS [29], relying on large-scale splitting strategy (upper),
leads to ill-position Gaussian growth in complex regions.
prevent effective parameter updates, creating persistent artifacts
that cannot be resolved merely through aggressive densification.
This frozen phenomenon explains why existing methods focusing
solely on densification criterion modifications still fail to fully ad-
dress over-reconstruction, despite employing more primitives [35].
Built upon these insights, we propose ReAct-GS, a method rooted
in the principle of re-activation to address over-reconstruction. To
address gradient magnitude dilution, we develop an importance-
aware densification criterion that re-activates stalled growth in
complex regions by incorporating 𝛼-blending weights from multi-
ple viewpoints into gradient aggregation during densification. To
eliminate the primitive frozen phenomenon, we introduce a novel
re-activation mechanism that revitalizes trapped primitives through
adaptive parameter perturbations, thereby expanding their percep-
tual range. While such perturbation might intuitively compromise
training stability, our experiments demonstrate that it successfully
re-activates frozen primitives for continued optimization, achieving
both efficient primitive utilization and effective artifact removal in
challenging scenarios.
We evaluate ReAct-GS across multiple public real-world datasets,
demonstrating significant improvements in both quantitative met-
rics and qualitative results. Our method achieves superior perfor-
mance on standard novel view synthesis metrics while retaining
comparable memory consumption to state-of-the-art approaches.
As shown in Figure 1 and Figure 2, our approach effectively resolves
the over-reconstruction problem while preserving geometrically ac-
curate structures, as evidenced by both rendered images and depth
maps. In summary, our contributions are as follows:
• We identify and analyze two fundamental limitations in
current 3D-GS pipelines: gradient magnitude dilution and
primitive frozen phenomenon, which collectively cause over-
reconstruction artifacts.
• We propose the ReAct-GS with a novel re-activation prin-
ciple, implemented by importance-aware densification and
adaptive re-activation mechanism, to address the limitations.
• Extensive experiments demonstrate that our method success-
fully eliminates over-reconstruction artifacts while achiev-
ing superior novel view synthesis quality and geometrically
precise reconstruction.

<!-- page 3 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
2
Related Works
Gaussian Splatting. 3D Gaussian Splatting (3D-GS) [10], a repre-
sentative point-based 3D rendering method, has garnered increas-
ing attention in the novel view synthesis domain. Unlike previous
structured implicit radiance fields, e.g., NeRF [19], 3D-GS pioneers
the reconstruction of 3D scenes using 3D elliptical Gaussian primi-
tives with learnable parameters. Through elaborate optimization
strategies coupled with parallelizable splat-style rasterization tech-
niques [38], 3D-GS has quickly emerged as a protagonist among
mainstream 3D neural representations due to its fast training and
computational efficiency and real-time high-fidelity rendering ca-
pabilities. Based on 3D-GS’s advanced characteristics, numerous
follow-up studies have explored enhancements and applications,
including alias-free rendering [31], efficient training [8, 18, 21],
editing and generation [3, 4, 30], etc. [5, 16, 22, 37]. While extended
topics built upon 3D-GS are under active exploration [2, 27, 36], its
critical over-reconstruction deficiency [10] remains unsolved and
impedes further deployment in complex real-world scenarios [23].
Over-reconstruction and densification. 3D-GS typically shows
textural degradation in representing high-frequency regions or fine-
grained repetitive textures, which is defined as over-reconstruction.
Essentially, this issue manifests when covering complex scenes with
only a small number of 3D Gaussian primitives that are difficult to
optimize [10], attributed to sparse point cloud initialization [24, 35]
and the inherent absence of periodicity features in Gaussian func-
tion [34]. The vanilla Adaptive Density Control (ADC) strategy [10]
addresses this by splitting primitives based on average gradient
magnitude under Normalized Device Coordinates (NDC). However,
this approach struggles with challenging conditions, particularly
unbounded outdoor scenes and tiny details. Recent studies pro-
pose various solutions to this limitation. Pixel-GS [35] introduces
dynamic pixel-aware gradients by incorporating projected pixel
numbers as weights to facilitate large-scale primitive splitting, while
Abs-GS [29] addresses gradient collision flaw through absolute op-
erator to encourage similar splitting behavior. Mini-Splatting [5]
takes a different approach by employing a threshold to identify and
actively split large-scale Gaussians that cause blur in rendering, aug-
menting the original gradient-based criterion. Additional improve-
ments include residual densification [17], smooth densification [11],
and visual consistent split [6], among others [12, 14, 28, 32].
Building upon prior works, we conduct a thorough analysis
of current gradient-based densification schemes and demonstrate
that the differential treatment of large-scale primitives in densi-
fication adversely affects the final 3D scene representation. Con-
sequently, we propose an advanced densification criterion that
incorporates importance-awareness. Furthermore, we identify the
primitive frozen phenomenon as another critical factor of over-
reconstruction artifacts and introduce an adaptive re-activation
mechanism to address these challenges.
3
Method
In this section, we first review the background of 3D-GS in Sec-
tion 3.1; then, we analyze the gradient magnitude dilution issue in
current gradient criterion and propose importance-aware densifi-
cation in Section 3.2; finally, we introduce the primitive frozen phe-
nomenon and re-activation mechanism to overcome such limitation
and effectively address over-reconstruction artifacts in Section 3.3.
3.1
Preliminaries
3D Gaussian representation. 3D-GS [10] fits a scene by optimiz-
ing a set of learnable 3D Gaussian primitives {𝐺𝑖| 𝑖= 1, · · · , 𝑁}. A
3D Gaussian 𝐺𝑖can be represented in 3D space as:
𝐺𝑖(x) = 𝑒−1
2 (x−𝝁3𝐷
𝑖
)𝑇(𝛴3𝐷
𝑖
)−1 (x−𝝁3𝐷
𝑖
),
(1)
where 𝝁3𝐷
𝑖
∈R3×1 and 𝛴3𝐷
𝑖
∈R3×3 represent the center position
and covariance matrix, respectively. Additionally, each Gaussian
primitive is also characterized with two extra attributes: opacity
𝑜𝑖∈[0, 1] and color c𝑖in spherical harmonic (SH) coefficients to
encode view-dependent color for rendering. During splatting, the
3D Gaussian primitive 𝐺𝑖is projected to 2D screen space by 6-DoF
transformation matrix and affine approximation, obtaining 𝐺2𝐷
𝑖
with 𝝁2𝐷
𝑖
and 𝛴2𝐷
𝑖
. With extensive number of primitives splatting
onto image plane, 3D-GS utilizes differentiable 𝛼-blending to render
the color for each pixel p in the depth order as:
c(p) =
𝑁
∑︁
𝑖=1
c𝑖𝜔𝑖,𝜔𝑖= 𝛼𝑖(p)
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗(p)), 𝛼𝑖(p) = 𝑜𝑖𝐺2𝐷
𝑖
(p), (2)
where 𝑁is the number of Gaussians involved in 𝛼-blending [38].
These 3D Gaussian parameters are optimized under multi-view
supervision through a composite photometric loss 𝐿.
Adaptive Density Control (ADC). Since Gaussians are initial-
ized from sparse point clouds produced from SfM [25], ADC strat-
egy is designed to densify 3D Gaussian primitives to better rep-
resent sparse areas. To address the over-reconstruction and under-
reconstruction, ADC conducts split or clone operation for large or
small Gaussian primitives, respectively. For a 3D Gaussian 𝐺𝑖and
its 2D image plane projection center 𝝁𝑘
𝑖= (𝝁𝑘
𝑖,𝑥, 𝝁𝑘
𝑖,𝑦) under the 𝑘
viewpoint, the densification will be performed if its average gradi-
ent ∇𝝁𝑖𝐿on screen space over 𝑀viewpoints satisfies
∇𝝁𝑖𝐿= 1
𝑀
𝑀
∑︁
𝑘=1
∇𝝁𝑖𝐿𝑘 = 1
𝑀
𝑀
∑︁
𝑘=1
v
u
t 
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
!2
+
 
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
!2
,
(3)
where 𝜏𝑝𝑜𝑠is a predefined threshold for recognizing large gradient
elements, and another hyper-parameter 𝜏𝑠𝑖𝑧𝑒is set to discriminate
large or small primitives by their largest principal scale:
∇𝝁𝑖𝐿> 𝜏𝑝𝑜𝑠and
 max(𝑠𝑥,𝑠𝑦,𝑠𝑧) > 𝜏𝑠𝑖𝑧𝑒,
split
max(𝑠𝑥,𝑠𝑦,𝑠𝑧) < 𝜏𝑠𝑖𝑧𝑒,
clone
(4)
3.2
Importance-Aware Densification
Despite advancements in addressing over-reconstruction, the un-
intended artifact in Figure 2, resulting from the large-scale 3D
Gaussian splitting technique [5, 29], indicates this problem remains
unsolved. Through an in-depth analysis of the average gradient
densification criterion, we identify a key factor–gradient magnitude
dilution–which causes densification to stall when fitting complex re-
gions by overlooking the primitive’s importance in 𝛼-blending ren-
dering. In response to this issue, we propose an importance-aware
densification criterion to re-activate the densification process.

<!-- page 4 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
(c) Re-activate by density-aware clone
(d) Re-activate by needle-shape perturbation
scale up to expand 
perceptual range   
needle-shape
few pixels 
contribute to 
gradient
gradient decay 
exponentially 
acts as background
but show over-construction
local optimal 
but unaware of 
neighbor flaws
clone to low 
dense area 
(b) Needle-shape frozen Gaussian
(a) Small-scale frozen Gaussian
Figure 3: Illustration of primitive frozen phenomenon and
our adaptive re-activation mechanism. (a) and (b) depict the
underlying factors causing small-scale and needle-shape 3D
Gaussians to freeze; (c) and (d) showcase our proposed re-
activation strategies for these respective frozen primitives.
For Gaussian 𝐺𝑖with projection center 𝝁𝑘
𝑖on 𝑘-th viewpoint, we
assume that it is splatted onto𝑚𝑘
𝑖pixels {p1, p2, . . . , p𝑚𝑘
𝑖}, obtaining
the gradient ∇𝝁𝑖𝐿𝑘of Equation (3) in NDC space as follows:
∇𝝁𝑖𝐿𝑘=
𝑚𝑘
𝑖
∑︁
𝑗=0
∇𝝁𝑖𝐿𝑘
𝑗=
©­­
«
𝑚𝑘
𝑖
∑︁
𝑗=0
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
,
𝑚𝑘
𝑖
∑︁
𝑗=0
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑦
ª®®
¬
.
(5)
Focusing on the 𝑥-axis component ∇𝝁𝑖𝐿𝑘
𝑗of the gradient with re-
spect to the 𝑗-th pixel, we derive the following analytical formula-
tion:
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
=
𝜕𝐿𝑘
𝑗
𝜕𝑐(p𝑗) × 𝜕𝑐(p𝑗)
𝜕𝛼𝑖(p𝑗) × 𝜕𝛼𝑖(p𝑗)
𝜕𝝁𝑘
𝑖,𝑥
(6)
The above discussion is simplified to consider only a single channel
of the RGB color. The term 𝜕𝐿𝑘
𝑗/𝜕𝑐(p𝑗) corresponds to the gradient
of the loss function employed in the optimization process. For the
remaining terms, we derive their detailed expressions according to
the 𝛼-blending formulation presented in Equation (2):
𝜕𝑐
𝜕𝛼𝑖
= 𝑐𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗) +
𝑁
∑︁
𝑙=𝑖+1
(−𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=1,𝑗≠𝑖
(1 −𝛼𝑗))
𝜕𝛼𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝑜𝑖× 𝐺2𝐷
𝑖
× 𝑔′
𝑖,𝑥,
𝜕𝐺2𝐷
𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝐺2𝐷
𝑖
× 𝑔′
𝑖,𝑥
(7)
where p𝑗is omitted for clarity, and𝑐and 𝛼represent the correspond-
ing color and alpha values for each Gaussian in depth order. The
term 𝜕𝐺2𝐷
𝑖
/𝜕𝝁𝑖,𝑥is expressed as 𝐺2𝐷
𝑖
× 𝑔′
𝑖,𝑥where 𝑔′
𝑖,𝑥is a complex
item dependent on the projected 2D covariance matrix of 𝐺𝑖and
the distance between p𝑗and 𝝁𝑘
𝑖. After combining and simplifying,
we derive the following equation (detailed in Appendix):
𝜕𝑐
𝜕𝛼𝑖
× 𝜕𝛼𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝜔𝑘
𝑖·
(
𝑐𝑖−
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙· 𝛼𝑙
𝑙−1
Ö
𝑗=𝑖+1
(1 −𝛼𝑗)
#)
· 𝑔′
𝑖,𝑥
(8)
where 𝜔𝑘
𝑖is 𝐺𝑖’s rendering weight defined in Equation (2). Equa-
tion (8) reveals that the gradient magnitude of 𝐺𝑖from the 𝑘-th
viewpoint is positively correlated with its importance in rendering.
When 𝐺𝑖is located in a complex area from the 𝑘-th viewpoint with
large 𝜔𝑘
𝑖(e.g., positioned earlier in depth order) but has low ren-
dering weights from other views (positioned later in depth order),
the average gradient magnitude in Equation (3) can fall below 𝜏𝑝𝑜𝑠,
stagnating its densification and leading to persistent blurring arti-
facts. Intuitively, gradients of 𝐺𝑖from viewpoints where it plays a
significant role (higher 𝜔𝑖) should have greater influence in deter-
mining whether densification is needed for optimal representation.
However, this limitation severely hinders the growth of model ca-
pacity to reconstruct complex scenes–an effect we term as gradient
magnitude dilution and identify as a fundamental cause of the
over-reconstruction problem.
Based on theoretical analysis, we propose a novel and principled
approach to re-activate stalled densification, termed importance-
aware densification. Considering the densification of 𝐺𝑖after opti-
mization through 𝑀𝑖views with {𝑚𝑘
𝑖|𝑘∈{1, . . . , 𝑀𝑖}} pixels splat-
ted for each view, we integrate the rendering importance weights
{𝜔𝑘
𝑖,𝑗|𝑗∈{1, . . . ,𝑚𝑘}} into the ∇𝝁𝑖𝐿calculation as:
∇𝝁𝑖𝐿=
Í𝑀
𝑘=1 𝜔𝑘
𝑖
√︄
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
2
+

𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
2
Í𝑀
𝑘=1 𝜔𝑘
𝑖
, 𝜔𝑘
𝑖=
Í𝑚𝑘
𝑖
𝑗=1 𝜔𝑘
𝑖,𝑗
𝑚𝑘
𝑖
(9)
where the Í𝑚𝑘
𝑖
𝑗=0 𝜔𝑘
𝑖,𝑗can be accumulated during the forward pass
without introducing additional computational overhead. Crucially,
we normalize the accumulated 𝜔𝑘
𝑖,𝑗values by the number of pixels
𝑚𝑘
𝑖that 𝐺𝑖splats onto each viewpoint, which effectively mitigates
densification bias towards Gaussians with larger projected areas.
This normalization strategy is essential for preventing inappropri-
ate splitting of large-scale Gaussians, as demonstrated in Figure 2.
Our comprehensive experiments demonstrate that the pro-
posed importance-aware densification criterion consistently out-
performs existing approaches in high-frequency texture represen-
tation by systematically cultivating critical primitives to restore
over-reconstruction regions, as thoroughly analyzed in Section 4.
3.3
Re-Activation Mechanism
Although enhanced densification strategies demonstrate substan-
tial improvement, they remain inadequate for completely address-
ing all over-reconstruction artifacts. Through rigorous analysis,
we identify another fundamental limitation: the primitive frozen
phenomenon, wherein established Gaussians become resistant to
subsequent optimization refinement.
Primitive frozen phenomenon. During 3D-GS training, 3D Gauss-
ian primitives rapidly converge to fit local textures within a few
iterations. However, we observe that this premature convergence
impedes continuous optimization of specific primitives. Combining
our analysis from Equation (8) with the 𝛼-blending formulation

<!-- page 5 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
Table 1: Quantitative results on Mip-NeRF 360[1], Tanks & Temples[13], and Deep Blending[9]. † indicates that the metrics are
directly sourced from the original 3D-GS report [10]. The scores are highlighted as: 1st- , 2nd- , and 3rd- best performances,
respectively. The MB and GB represent Megabyte and Gigabyte in storage.
Datasets
Mip-NeRF 360
Tanks & Temples
Deep Blending
Methods
PSNR ↑
SSIM ↑
LPIPS ↓
Mem
PSNR ↑
SSIM ↑
LPIPS ↓
Mem
PSNR ↑
SSIM ↑
LPIPS ↓
Mem
Plenoxels † [7]
23.08
0.626
0.463
2.1GB
21.08
0.719
0.379
2.3GB
23.06
0.795
0.510
2.7GB
INGP-Big † [20]
25.59
0.699
0.331
48MB
21.92
0.745
0.305
48MB
24.96
0.817
0.390
48MB
Mip-NeRF360 † [1]
27.69
0.792
0.237
8.6MB
22.22
0.759
0.257
8.6MB
29.40
0.901
0.245
8.6MB
3D-GS † [10]
27.21
0.815
0.214
734MB
23.14
0.841
0.183
411MB
29.41
0.903
0.243
676MB
3D-GS [10]
27.29
0.808
0.225
793MB
23.61
0.853
0.173
347MB
29.41
0.907
0.195
515MB
Blur-split [5]
27.11
0.810
0.208
634MB
22.98
0.844
0.158
638MB
29.09
0.898
0.194
950MB
Abs-GS [29]
27.27
0.816
0.196
678MB
23.39
0.857
0.161
417MB
29.36
0.907
0.190
587MB
Pixel-GS [35]
27.72
0.832
0.178
1145MB
23.75
0.862
0.149
708MB
29.49
0.907
0.191
760MB
ReAct-GS (ours)
27.79
0.835
0.176
805MB
24.06
0.865
0.145
787MB
29.66
0.908
0.191
585MB
15k
20k
30k
Final
Final
15k
7k
5k
Needle-shape
Small-scale
Figure 4: Visualization of primitive frozen phenomenon
across optimization stages. Top: Small-scale 3D Gaussians
freeze locally and barely exhibit growth to relieve blurry dur-
ing densification; Bottom: Needle-shape primitives persist
and cannot be mitigated in post-densification optimization.
in Equation (2), the gradient magnitude of primitives is related to
𝛼𝑘
𝑖,𝑗, which can be decomposed as:
𝛼𝑘
𝑖,𝑗= 𝑜𝑖× 𝑒−1
2 (p𝑗−𝝁𝑘
𝑖)𝑇𝛴𝑖,𝑘
2𝐷(p𝑗−𝝁𝑘
𝑖)
(10)
This formulation reveals that gradient magnitude ∇𝝁𝑖𝐿𝑘
𝑗diminishes
exponentially with increasing distance between pixel p𝑗and pro-
jection center 𝝁𝑘
𝑖. As noted in Pixel-GS [35], only pixels in close
proximity to the projection center 𝝁𝑘
𝑖contribute substantially to the
gradient of these Gaussian primitives. We propose that this intrinsic
characteristic fundamentally restricts small-scale and needle-shape
primitives from optimization beyond their initial convergence state.
In Figure 3(a), a small-scale primitive converges to fit local con-
tent with minimal pixel contribution to its gradients. Due to its
limited spatial perception, this primitive cannot adequately expand
to cover blurry regions, perpetuating over-reconstruction artifacts.
Similarly, the needle-shape primitive illustrated in Figure 3(b) ex-
hibits constrained optimization: its short axis is influenced by only a
few adjacent pixels while gradient magnitude exponentially decays
along the long axis, severely limiting its capacity to deform under
standard optimization procedures.
We empirically visualize the evolution of 3D Gaussians to verify
our analysis, as shown in Figure 4. We render small-scale and needle-
shape Gaussians in densification and post-densification stages, re-
spectively. Theoretically, small-scale Gaussians should spread to
cover blurry regions and represent fine details. However, as evident
in the upper section of Figure 4, the distribution of these Gaussians
exhibits minimal migration toward over-reconstruction regions
between 5k and 15k iterations, demonstrating localized freezing
behavior. Similarly, needle-shape Gaussians remain virtually un-
changed from 15k iterations onward. The lower section of Figure 4
demonstrates that these needle-shape artifacts emerge at the post-
densification stage’s commencement and persist throughout train-
ing completion, indicating the inability of existing optimization
methods to resolve these elongated structures.
To revitalize these frozen primitives for effective optimiza-
tion, we propose a re-activation mechanism consisting of Density-
Guided Clone and Needle-Shape Perturbation.
Density-Guided Clone. In contrast to exact primitive replication
prevalent in conventional densification, we propose strategically
relocating cloned primitives based on local density to populate
sparse regions, as illustrated in Figure 3(c). For a 3D Gaussian 𝐺𝑖
with ∇𝝁𝑖𝐿exceeding the densification threshold, we quantify its
local density 𝑑𝑖through the K-nearest-neighbors algorithm:
𝑑𝑖= 1
𝐾
∑︁
𝐺𝑗∈𝑁𝐾(𝐺𝑖)
∥𝝁𝑖−𝝁𝑗∥
(11)
where 𝝁𝑗represents the center of 𝐺𝑗, and 𝐾= 3 balances accuracy
and computational efficiency. Utilizing this density metric, we po-
sition the cloned primitive 𝐺′
𝑖by sampling from the distribution
N (𝝁𝑖,𝑑𝑖· I3). This density-guided approach confers sufficient opti-
mization momentum to primitives in sparse regions, enabling them
to overcome local optima and respond to neighboring gradients.
The mechanism facilitates small Gaussians’ migration toward over-
reconstruction areas while negligibly perturbing primitives already
situated within dense clusters. Our experimental results in Figure 7
demonstrate the enhanced detail reconstruction achieved through
this approach, particularly when collaborating with our proposed
importance-aware densification.
Needle-Shape Perturbation. While the aforementioned techniques
deliver satisfactory performance in complex scene reconstruction,
needle-shape primitives continue to introduce visual artifacts in
challenging corner-case renderings. These elongated Gaussians

<!-- page 6 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
3D-GS
Abs-GS
Pixel-GS
ReAct-GS (ours)
GT
train
flowers
garden
stump
room
bonsai
Figure 5: Qualitative comparisons of different methods on scenes from the Mip-NeRF 360 and Tanks & Temples datasets. Close-
up views highlight the challenging areas with high-frequency details, where over-reconstruction is particularly pronounced.
typically do not satisfy clone criteria due to their dominant principal
axis. We therefore propose perturbing their shape to expand their
perceptual range, as illustrated in Figure 3(d). Specifically, for 3D
Gaussians 𝐺𝑖with scale values 𝑠𝑖= (𝑠1
𝑖,𝑠2
𝑖,𝑠3
𝑖), we identify needle-
shape primitives using the criterion:
𝐺𝑛𝑠=

𝐺𝑖| max{𝑠𝑖}
Í𝑠𝑖
> 𝜏𝑛𝑠,𝑖∈{1, . . . , 𝑁}

(12)
where 𝜏𝑛𝑠is a threshold set to 0.8 in our implementation. For
these identified primitives, we perturb their shape by magnify-
ing the shorter principal axes by a factor 1
2𝑠𝑑𝑒𝑔where 𝑠𝑑𝑒𝑔=
max{𝑠𝑖}/mid{𝑠𝑖}. This shape perturbation introduces additional
neighboring pixels with significant gradient contribution, facili-
tating needle-shape Gaussian optimization. Analogous to opacity
reset in conventional 3D-GS, we apply needle-shape perturbation
periodically at intervals of 3k iterations to minimize disruption to
the optimization process. Our experiments demonstrate that this ap-
proach effectively eliminates most needle-like artifacts, particularly
in rendering corner regions, as detailed in Section 4.2.
4
Experiments
4.1
Setup
Datasets. Following the 3D-GS [10] evaluation protocol, we as-
sess ReAct-GS on 13 diverse real-world scenes spanning indoor
enclosed spaces and outdoor areas. Specifically, we employ: (1) all
9 indoor/outdoor scenes from Mip-NeRF360 [1], (2) two additional
outdoor environments (Train and Truck) from [13], and (3) two
indoor scenes (drjohnson and playroom) from [9]. This selection

<!-- page 7 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
GT rendering
ReAct-GS
Pixel-GS
Abs-GS
3D-GS
room
stump
treehill
counter
Figure 6: Comparison of depth map renderings across different methods on indoor/outdoor scenes from the Mip-NeRF 360
dataset. Enlarged views highlight the challenging areas where geometric consistency and coherence are crucial.
ensures a thorough evaluation across varied settings while main-
taining consistency with prior works.
Baselines. We perform comprehensive comparisons against ad-
vanced NeRF-based methods [1, 7, 20] and recent Gaussian Splat-
ting methods (3D-GS [10] and its variants designed to address over-
reconstruction issues [5, 29, 35]). For NeRF-based approaches, we
utilize the quantitative metrics from the original publications [10].
For GS-based approaches, we use official implementations with
consistent configurations, disabling auxiliary strategies that might
affect fair comparison. Regarding Mini-Splatting [5], we focus on
its core Blur-split module (which targets over-reconstruction mit-
igation) while omitting other components (depth-reinitialize and
pruning), labeled as "Blur-split" in our results for clarity.
Implementation details. All experiments were conducted on an
NVIDIA RTX-3090 GPU with 24 GB of memory. Following standard
Gaussian splatting protocols, we adopt the progressive densifica-
tion schedule, densifying the Gaussian primitives from 500 to 15k
at 100-iteration intervals, with training termination at 30k itera-
tions. We maintain the original 3D-GS evaluation with consecutive
8-frame training segments and fixed test frames. Our evaluation
reports standard novel view synthesis metrics (PSNR, SSIM, LPIPS)
along with memory consumption for well-optimized Gaussian pa-
rameter storage. To address over-reconstruction, ReAct-GS employs
importance-aware densification during both clone and split oper-
ations. This approach inherently amplifies gradient magnitudes
during optimization, necessitating an adjusted gradient threshold
(𝜏𝑝𝑜𝑠= 3𝑒−4) to ensure memory constraints and stable training. For
baseline comparisons, we faithfully reproduce their original con-
figurations: 𝜏𝑝𝑜𝑠= 4𝑒−4 for Abs-GS split operation and 𝜏𝑝𝑜𝑠= 2𝑒−4
for other methods, as specified in their respective publications.
4.2
Main Results
Quantitative results. The quantitative comparison across vari-
ous methods is presented in Table 1. It is noteworthy that ReAct-
GS demonstrates consistent superiority across all evaluation met-
rics, validating the effectiveness of our proposed importance-aware
densification and novel re-activation mechanism. While ReAct-GS
maintains slightly higher parameter counts (increased memory
for storage) than methods focusing exclusively on splitting large-
scale primitives (Blur-split and Abs-GS), our method eliminates the
depth inconsistency artifacts (shown in Figure 2) that plague split-
intensive approaches, which is a crucial factor for 3D reconstruction.
Notably, ReAct-GS achieves these results with fewer primitives than
Pixel-GS, while simultaneously delivering superior performance
in both over-reconstruction mitigation and fine detail preserva-
tion, demonstrating ReAct-GS’s exceptional efficiency in primitive
utilization.
Qualitative results. In Figure 5, we compare novel view render-
ings with state-of-the-art approaches. In addition to significantly
reducing the over-reconstruction blur and improving overall visual
quality across all scenes, our method also excels in reconstruct-
ing fine-grained textures in challenging regions, such as grassy

<!-- page 8 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
areas at the frame’s corners (flowers), dense foliage inside the tree
stump (stump), and object edges (room). To further assess geomet-
ric reconstruction capability, Figure 6 presents depth maps ren-
dered with the official 𝛼-blending variant [10]. ReAct-GS produces
depth-consistent reconstructions that comply with real-world ge-
ometry without spurious floaters, even in complex scenarios like
uneven gravel ground (treehill). Moreover, ReAct-GS outperforms
all competitors in distant object reconstruction, as evidenced by
precise depth outlines (e.g., geo-grid in room). These improvements
stem from our two key designs: (1) The importance-aware den-
sification strategically grows primitives close to object surfaces
where needed, enhancing detail without overfitting or disrupting
Gaussian distributions; (2) our re-activation mechanism mitigates
over-reconstruction and needle artifacts by recycling frozen Gaus-
sians for global optimization, avoiding parameter inflation while
encouraging uniform primitive distribution, thus improving effi-
ciency.
4.3
Ablation Study
In this section, we rigorously evaluate the efficacy of our proposed
modules through ablation experiments with quantitative results
presented in Table 2. Built upon the vanilla 3D-GS baseline, we ex-
plore four key component combinations. The results demonstrate
that our importance-aware densification drives substantial render-
ing quality improvements (+0.36 dB PSNR avg.) by reviving stalled
3D Gaussian growth to mitigate the persistent over-reconstruction
issue. Additionally, the re-activation mechanism effectively resolves
the remaining artifacts in challenging regions that existing methods
fail to address. Meanwhile, the density-guided clone also operates as
a standalone enhancement, improving detail representation even on
baseline 3D-GS. Complementing these metrics, Figure 7 provides vi-
sual evidence of each module’s necessity, particularly in texture-rich
areas where other approaches struggle. The progressive improve-
ments validate our hierarchical design: while importance-aware
densification establishes the foundation, re-activation delivers criti-
cal refinements for comprehensive scene representation.
To further validate the generalizability of our approach, we inves-
tigate the integration potential of our re-activation mechanism with
existing methods. Specifically, we select Pixel-GS, which represents
the state-of-the-art baseline with its own densification criterion
yet still exhibiting over-reconstruction in challenging regions (as
shown in the lower part of Figure 7). Remarkably, when augmented
with our re-activation module, Pixel-GS demonstrates complete
elimination of its persistent artifacts, as visually confirmed in Fig-
ure 7. These improvements are further quantified in the last row
of Table 2, where the hybrid approach achieves superior metrics
compared to standalone Pixel-GS. These results underscore that
our re-activation mechanism maintains effectiveness even when
transferred to alternative architectures, and that frozen primitive
remains a key factor in leading to over-reconstruction even when
advanced densification is employed.
4.4
Efficiency Analysis
Remarkably, ReAct-GS maintains excellent parameter efficiency
while enhancing performance. As demonstrated in Table 1, our ap-
proach yields significant quality enhancements without significant
Table 2: Ablation study on the Mip-NeRF 360 dataset. Re-
sults show average evaluation metrics, where IAD, DGC, and
NSP refer to importance-aware densification, density-guided
clone, and needle-shape perturbation, respectively.
PSNR ↑
SSIM ↑
LPIPS ↓
Mem
Baseline (3D-GS)
27.29
0.815
0.225
793MB
+ IAD
27.65
0.827
0.181
739MB
+ DGC
27.33
0.815
0.221
832MB
+ IAD + NSP
27.67
0.828
0.181
766MB
+ IAD + DGC
27.76
0.832
0.179
810MB
Full-equipped (ReAct-GS)
27.79
0.835
0.176
805MB
Pixel-GS [35] + re-activation
27.74
0.833
0.180
1191MB
3D-GS
w/ DGC
w/ IAD
w/ IAD & DGC 
ReAct-GS
Pixel-GS
3D-GS
w/ DGC
w/ IAD
w/ IAD & DGC 
Pixel-GS
ReAct-GS
Pixel-GS
Pixel-GS
Pixel-GS
w/ re-activation
w/ re-activation
w/ re-activation
Figure 7: Visualization of ablation experiments. Top: Visual
improvements from each proposed module on challenging
outdoor scenes. Bottom: Performance enhancement of Pixel-
GS when integrated with our re-activation mechanism.
parameter expansion. On the Mip-NeRF 360 dataset, our method re-
duces parameters by 29.7% (avg. training time of 33 mins on vanilla
3D-GS engine) while outperforming Pixel-GS (avg. training time of
41 mins) both quantitatively and qualitatively. Notably, Pixel-GS
represents the state-of-the-art method with minimal geometric in-
consistency artifacts. Although our approach requires marginally
increased computational resources compared to 3D-GS (avg. 26
mins for training) and Abs-GS (avg. 27 mins for training), we effec-
tively resolve the over-reconstruction phenomenon inherent in 3D-
GS and deliver more precise geometric reconstruction than Abs-GS,
which are critical prerequisites for practical 3D reconstruction ap-
plications. Furthermore, the computational costs could be reduced
through prospective pruning and acceleration techniques once re-
construction quality ceases to be a limiting factor. We attribute our

<!-- page 9 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
efficiency advancement to: (1) targeted densification, wherein our
importance-aware criterion strategically densifies 3D Gaussians in
critical over-reconstructed regions for visual quality (corroborated
by Table 2), thus avoiding superfluous additions of minimally con-
tributing primitives; (2) primitive recycling, where our re-activation
mechanism repurposes frozen Gaussians for beneficial optimization
rather than indefinitely introducing new elements.
5
Conclusion
In this work, we identify the critical underlying factors of over-
reconstruction in 3D-GS, particularly the deficiencies of gradient
magnitude dilution in current densification strategies and the previ-
ously overlooked primitive frozen phenomenon. These limitations
lead to persistent artifacts such as blurry textures and needle-shape
distortions, even when improved densification criteria are applied.
To enhance fine-detail reconstruction in complex scenes, we intro-
duce ReAct-GS equipped with two key modules: importance-aware
densification and an adaptive re-activation mechanism (comprising
density-guided clone and needle-shape perturbation) to re-activate
stalled primitive growth and frozen primitives for continual global
optimization. Benefiting from theoretical analysis, ReAct-GS effec-
tively combats over-reconstruction while inherently preventing ge-
ometrical inconsistencies. Comprehensive experiments, including
both quantitative and qualitative evaluations alongside thorough
ablation studies, demonstrate our approach’s significant improve-
ments in addressing over-reconstruction. Furthermore, ReAct-GS
maintains high parameter efficiency and training speed while ex-
hibiting strong transferability, consistently boosting performance
when integrated with other 3D-GS variants.
Acknowledgments
This research was supported by the Theme-based Research Scheme
(TRS) project T45-701/22-R of the Research Grants Council (RGC),
Hong Kong SAR. We thank all anonymous reviewers for their con-
structive feedback to improve our paper.
References
[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo
Martin-Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale repre-
sentation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF
international conference on computer vision. IEEE, 5855–5864.
[2] Yuanhao Cai, Yixun Liang, Jiahao Wang, Angtian Wang, Yulun Zhang, Xiaokang
Yang, Zongwei Zhou, and Alan Yuille. 2024. Radiative gaussian splatting for
efficient x-ray novel view synthesis. In European Conference on Computer Vision.
Springer, 283–299.
[3] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang,
Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. 2024. Gaussianeditor:
Swift and controllable 3d editing with gaussian splatting. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. IEEE, 21476–
21485.
[4] Yuxin Cheng, Binxiao Huang, Taiqiang Wu, Wenyong Zhou, Chenchen Ding,
Zhengwu Liu, Graziano Chesi, and Ngai Wong. 2025. Perspective-aware 3D
Gaussian Inpainting with Multi-view Consistency. arXiv preprint arXiv:2510.10993
(2025).
[5] Guangchi Fang and Bing Wang. 2024. Mini-splatting: Representing scenes with
a constrained number of gaussians. In European Conference on Computer Vision.
Springer, 165–181.
[6] Qiyuan Feng, Gengchen Cao, Haoxiang Chen, Tai-Jiang Mu, Ralph R Mar-
tin, and Shi-Min Hu. 2024. A new split algorithm for 3D Gaussian splatting.
arXiv:2403.09143 [cs.CVPR]
[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht,
and Angjoo Kanazawa. 2022. Plenoxels: Radiance Fields without Neural Networks.
In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
IEEE, 5491–5500. doi:10.1109/CVPR52688.2022.00542
[8] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker,
and Tom Goldstein. 2025. Pup 3d-gs: Principled uncertainty pruning for 3d
gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. IEEE, 5949–5958.
[9] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis,
and Gabriel Brostow. 2018. Deep blending for free-viewpoint image-based ren-
dering. ACM Transactions on Graphics (ToG) 37, 6 (2018), 1–15.
[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
2023. 3d gaussian splatting for real-time radiance field rendering. ACM Transac-
tions on Graphics 42, 4 (2023), 139–1.
[11] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che
Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi.
2024. 3d gaussian splatting as markov chain monte carlo. Advances in Neural
Information Processing Systems 37 (2024), 80965–80986.
[12] Sieun Kim, Kyungjin Lee, and Youngki Lee. 2024. Color-cued Efficient Densifica-
tion Method for 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR) Workshops. IEEE, 775–783.
[13] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and
temples: Benchmarking large-scale scene reconstruction. ACM Transactions on
Graphics (ToG) 36, 4 (2017), 1–13.
[14] Chai-Rong Lee, Ting-Yu Yen, Kai-Wen Hsiao, Shih-Hsuan Hung, Sheng-Chi Hsu,
Min-Chun Hu, Chih-Yuan Yao, and Hung-Kuo Chu. 2024. ODA-GS: Occlusion-
and Distortion-aware Gaussian Splatting for Indoor Scene Reconstruction. In
SIGGRAPH Asia 2024 Technical Communications. ACM New York, NY, USA, New
York, NY, USA, 1–4.
[15] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong
Tang. 2024. Manigaussian: Dynamic gaussian splatting for multi-task robotic
manipulation. In European Conference on Computer Vision. Springer, 349–366.
[16] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo
Dai. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
IEEE, 20654–20664.
[17] Yanzhe Lyu, Kai Cheng, Xin Kang, and Xuejin Chen. 2024. ResGS: Residual Densi-
fication of 3D Gaussian for Efficient Detail Recovery. arXiv:2412.07494 [cs.CVPR]
[18] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger,
Francisco Vicente Carrasco, and Fernando De La Torre. 2024. Taming 3dgs: High-
quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference
Papers. ACM New York, NY, USA, New York, NY, USA, 1–11.
[19] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi
Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance
fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
[20] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. In-
stant neural graphics primitives with a multiresolution hash encoding. ACM
transactions on graphics (TOG) 41, 4 (2022), 1–15.
[21] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and
Hamed Pirsiavash. 2024. Compgs: Smaller and faster gaussian splatting with
vector quantization. In European Conference on Computer Vision. Springer, 330–
349.
[22] Aashish Rai, Dilin Wang, Mihir Jain, Nikolaos Sarafianos, Kefan Chen, Srinath
Sridhar, and Aayush Prakash. 2025. Uvgs: Reimagining unstructured 3d gauss-
ian splatting using uv mapping. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. IEEE, 5927–5937.
[23] Samuel Rota Bulò, Lorenzo Porzi, and Peter Kontschieder. 2024. Revising densifi-
cation in gaussian splatting. In European Conference on Computer Vision. Springer,
347–362.
[24] Johannes Lutz Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion
Revisited. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. IEEE, 4104–4113.
[25] Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion
revisited. In Proceedings of the IEEE conference on computer vision and pattern
recognition. IEEE, 4104–4113.
[26] Xuechang Tu, Bernhard Kerbl, and Fernando de la Torre. 2024. Fast and Robust
3D Gaussian Splatting for Virtual Reality. In SIGGRAPH Asia 2024 Posters (SA
’24). ACM New York, NY, USA, New York, NY, USA, Article 43, 3 pages. doi:10.
1145/3681756.3697947
[27] Yuxuan Wang, Xuanyu Yi, Zike Wu, Na Zhao, Long Chen, and Hanwang Zhang.
2024. View-consistent 3d editing with gaussian splatting. In European Conference
on Computer Vision. Springer, 404–420.
[28] Zhanke Wang, Guanhua Wu, Zhiyan Wang, Lu Xiao, Runling Liu, Jiahao Wu, and
Ronggang Wang. 2025. HDA-GS: Hierarchical Density-Controlled for Anisotropic
3D Gaussian Splatting. In ICASSP 2025-2025 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1–5.
[29] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. 2024. Absgs:
Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM
International Conference on Multimedia. ACM New York, NY, USA, New York,
NY, USA, 1053–1061.
[30] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang,
Wenyu Liu, Qi Tian, and Xinggang Wang. 2024. Gaussiandreamer: Fast generation

<!-- page 10 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
from text to 3d gaussians by bridging 2d and 3d diffusion models. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE,
6796–6807.
[31] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024.
Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. IEEE, 19447–19456.
[32] Ding Yuan, Sizhe Zhang, Hong Zhang, Yangyan Deng, and Yifan Yang. 2025.
EMA-GS: Improving sparse point cloud rendering with EMA gradient and anchor
upsampling. Image and Vision Computing 154 (2025), 105433.
[33] Hongjia Zhai, Xiyu Zhang, Boming Zhao, Hai Li, Yijia He, Zhaopeng Cui, Hujun
Bao, and Guofeng Zhang. 2025. Splatloc: 3d gaussian splatting-based visual local-
ization for augmented reality. IEEE Transactions on Visualization and Computer
Graphics 31, 5 (2025), 3591–3601.
[34] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric Xing. 2024. Fregs:
3d gaussian splatting with progressive frequency regularization. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE,
21424–21433.
[35] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Hengshuang Zhao. 2024.
Pixel-gs: Density control with pixel-aware gradient for 3d gaussian splatting. In
European Conference on Computer Vision. Springer, 326–342.
[36] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-
Hsuan Yang. 2024. Drivinggaussian: Composite gaussian splatting for surround-
ing dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. IEEE, 21634–21643.
[37] Liyuan Zhu, Yue Li, Erik Sandström, Shengyu Huang, Konrad Schindler, and Iro
Armeni. 2025. LoopSplat: Loop Closure by Registering 3D Gaussian Splats. In
International Conference on 3D Vision (3DV). IEEE, 1–12.
[38] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. 2001.
EWA volume splatting. In Proceedings Visualization, 2001. VIS’01. IEEE, 29–538.

<!-- page 11 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
A
PRILIMINARIES–ADC VARIANTS.
The vanilla ADC is argued to be insufficient to break through over-
reconstruction dilemma and corresponding variants are promoted.
First, the Mini-Splatting introduces blur-split approach to di-
rect split 3D Gaussians with large influential projected area, i.e.,
𝐺blur, apart from the original gradient-based densification, shown
as follows:
𝐺blur = {𝐺𝑖|𝑆𝑖> 𝜏𝑏𝑙𝑢𝑟,𝑖∈(1, . . . , 𝑁)}
(13)
where 𝜏𝑏𝑙𝑢𝑟is a hyper-parameters and 𝑆𝑖is the number of pixels
where 𝐺𝑖is of largest weight in alpha blending.
From another aspect, the pixel-aware weighted average gradients
from different views is published in Pixel-GS as follows:
∇𝝁𝑖𝐿Pixel-GS =
Í𝑀
𝑘=1 𝑚𝑘
𝑖· 𝑓(𝑖,𝑘) ·
√︄
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
2
+

𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
2
Í𝑀
𝑘=1 𝑚𝑘
𝑖
(14)
where 𝑚𝑘
𝑖is the number of projected pixels of 𝐺𝑖under 𝑘-th view-
point and 𝑓(𝑖,𝑘) is a factor to suppress floaters close to camera.
The ∇𝝁𝑖𝐿pixel magnifies the split possibility of large scale Gaussians
initialized within sparse regions.
Third, the gradient collision phenomenon is discovered by Abs-
GS. Focused on 𝑥-axis component of ∇𝝁𝑖𝐿, the 𝜕𝐿𝑘/𝜕𝝁𝑘
𝑖,𝑥is sum of
each independent gradient with respect to loss on pixel 𝑗affected
by 𝐺𝑖. As shown at left part of Equation (15), the hetero-directional
gradients cancel each other and reduce the final magnitude when
𝑚𝑘
𝑖goes larger, which hinders the large scale Gaussians densifica-
tion. Subsequently, the homo-directional gradient is proposed as
shown at right part as follows:
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
=
𝑚𝑘
𝑖
∑︁
𝑗=1
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
→
 
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
!
Abs-GS
=
𝑚𝑘
𝑖
∑︁
𝑗=1

𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥

(15)
The same operation is performed for 𝑦-axis as well. With the ab-
solute operation on sub-gradients of each pixel 𝑗, the collision
phenomenon is eliminated to restore split on large primitives.
B
GRADIENT DECOMPOSITION ANALYSIS
Based on the decomposition of ∇𝝁𝑖𝐿𝑘
𝑗in main part, shown as fol-
lows:
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
=
𝜕𝐿𝑘
𝑗
𝜕𝑐(p𝑗) × 𝜕𝑐(p𝑗)
𝜕𝛼𝑖(p𝑗) × 𝜕𝛼𝑖(p𝑗)
𝜕𝝁𝑘
𝑖,𝑥
(16)
Gradient magnitude dilution. The first term 𝜕𝐿𝑘
𝑗/𝜕𝑐(p𝑗) in
above equation is only related to the loss function employed for
training and irrelated with Gaussian primitives’ parameters. There-
fore, we can ignore it in following analysis. For clarity, we only
focus on 𝑥-axis component and a single channel of rendering color
in following analysis and similar conclusion can be drawn on𝑦-axis
component and remain two channels of rendering color. We can
decompose the last two term of Equation (16) as follows (same as
main part):
𝜕𝑐
𝜕𝛼𝑖
= 𝑐𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗) +
𝑁
∑︁
𝑙=𝑖+1
(−𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=1,𝑗≠𝑖
(1 −𝛼𝑗))
𝜕𝛼𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝑜𝑖× 𝐺2𝐷
𝑖
× 𝑔′
𝑖,𝑥,
𝜕𝐺2𝐷
𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝐺2𝐷
𝑖
× 𝑔′
𝑖,𝑥
(17)
Combined with above two equations, we can derive the
𝜕𝑐(p𝑗)
𝜕𝛼𝑖(p𝑗) ×
𝜕𝛼𝑖(p𝑗)
𝜕𝝁𝑘
𝑖,𝑥
as follows:
𝜕𝑐
𝜕𝛼𝑖
× 𝜕𝛼𝑖
𝜕𝝁𝑘
𝑖,𝑥
= 𝑔′
𝑥
(
𝑐𝑖𝑜𝑖𝐺2𝑑
𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗)
+𝑜𝑖𝐺2𝑑
𝑖
𝑁
∑︁
𝑙=𝑖+1
"
(−1)𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=1,𝑗≠𝑖
(1 −𝛼𝑗)
#)
= 𝑔′
𝑥
(
𝑐𝑖𝛼𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗) −𝛼𝑖
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=1,𝑗≠𝑖
(1 −𝛼𝑗)
#)
= 𝑔′
𝑥
(
𝑐𝑖𝜔𝑖−𝛼𝑖
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=1,𝑗≠𝑖
(1 −𝛼𝑗)
#)
= 𝑔′
𝑥
(
𝑐𝑖𝜔𝑖−𝛼𝑖
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙𝛼𝑙
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗)
𝑙−1
Ö
𝑗=𝑖+1
(1 −𝛼𝑗)
#)
= 𝑔′
𝑥
(
𝑐𝑖𝜔𝑖−𝛼𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗)
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=𝑖+1
(1 −𝛼𝑗)
#)
= 𝑔′
𝑥
(
𝑐𝑖𝜔𝑖−𝜔𝑖
𝑁
∑︁
𝑙=𝑖+1
"
𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=𝑖+1
(1 −𝛼𝑗)
#)
= 𝜔𝑖·
(
𝑐𝑖−
𝑁
∑︁
𝑙=𝑖+1
"
·𝑐𝑙𝛼𝑙
𝑙−1
Ö
𝑗=𝑖+1
(1 −𝛼𝑗)
#)
· 𝑔′
𝑥
(18)
Based on the above derivation, we can verify that 𝜕𝐿𝑘
𝑗/𝜕𝝁𝑘
𝑖,𝑥
is positively correlated with the rendering importance score 𝜔𝑘
𝑖.
Therefore, overlooking 𝜔𝑘
𝑖in ∇𝝁𝑖𝐿potentially causes gradient dilu-
tion. For a Gaussian primitive 𝐺𝑖with high weight 𝜔𝑘
𝑖from the 𝑘-th
viewpoint, 𝐺𝑖likely appears earlier in the corresponding depth or-
der. Consequently,𝐺𝑖may be occluded by other Gaussian primitives
𝐺𝑗from different viewpoints, where 𝐺𝑗plays a more significant
role (located earlier in depth order), resulting in 𝐺𝑖having smaller
rendering weights in these viewpoints. Due to these low rendering
weights leading to low gradient magnitudes, the overall average
gradient magnitude of 𝐺𝑖may fall below the threshold 𝜏𝑝𝑜𝑠, as its
high gradients from important viewpoints are diluted by numerous
low gradients from less influential viewpoints. Under these condi-
tions, 𝐺𝑖will not be densified despite being located within complex
regions that require more primitives to represent fine details. Our
experiments confirm that this factor significantly impairs current
3D-GS optimization and contributes to blurring artifacts.
Importance-aware densification. To overcome the gradient mag-
nitude dilution, we propose the importance-aware densification
criterion, which takes the rendering importance into account when
normalizing the gradients from multiple viewpoints after several
optimization iterations. For each pixel p𝑘
𝑗the 𝐺𝑖is splatted onto

<!-- page 12 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
from 𝑘viewpoint, we need to multiply corresponding rendering
weight 𝜔𝑘
𝑖,𝑗with it as follows:
∇𝝁𝑖𝐿𝑘
𝑗= 𝜔𝑘
𝑖,𝑗
 𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
,
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑦
!
,
(19)
If 𝐺𝑖is totally splatted on to 𝑚𝑘
𝑖pixels {p𝑘
𝑗, 𝑗∈[1, . . . ,𝑚𝑘
𝑖]} in the
𝑘-th viewpoint, we need to aggregate all importance-aware ∇𝝁𝑖𝐿𝑘
𝑗
as the NDC gradient of𝐺𝑖(consistent with 3D-GS rendering engine)
as follows:
∇𝝁𝑖𝐿𝑘=
©­­
«
𝑚𝑘
𝑖
∑︁
𝑗
𝜔𝑘
𝑖,𝑗·
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
,
𝑚𝑘
𝑖
∑︁
𝑗
𝜔𝑘
𝑖,𝑗·
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑦
ª®®
¬
(20)
which leads to additional memory consumption and computational
overhead in parallel CUDA kernel. Therefore, we simplify the above
formulation as follows:
∇𝝁𝑖𝐿𝑘= 𝜔𝑘
𝑖·
©­­
«
𝑚𝑘
𝑖
∑︁
𝑗
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
,
𝑚𝑘
𝑖
∑︁
𝑗
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑦
ª®®
¬
, 𝜔𝑘
𝑖=
Í𝑚𝑘
𝑖
𝑗
𝜔𝑘
𝑖,𝑗
𝑚𝑘
𝑖
(21)
Notably, the 𝜔𝑘
𝑖can be easily accumulated and calculated with only
one additional float memory consumption and one additional int
memory to record 𝑚𝑘
𝑖. Meanwhile, the normalization of 𝜔𝑘
𝑖over
projection area 𝑚𝑘
𝑖also eliminates the bias caused by large-scale
Gaussians being split aggressively, thus preventing the geometric
artifacts illustrated in main part. Based on the new importance-
aware ∇𝝁𝑖𝐿𝑘, we can obtain importance-aware ∥∇𝝁𝑖𝐿𝑘∥as:
∥∇𝝁𝑖𝐿𝑘∥= 𝜔𝑘
𝑖·
v
u
t 
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
!2
+
 
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
!2
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
=
𝑚𝑘
𝑖
∑︁
𝑗
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑥
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
=
𝑚𝑘
𝑖
∑︁
𝑗
𝜕𝐿𝑘
𝑗
𝜕𝝁𝑘
𝑖,𝑦
(22)
and then we redefine the importance-aware densification criterion
∇𝝁𝑖𝐿as:
∇𝝁𝑖𝐿=
Í𝑀
𝑘=1 ∥∇𝝁𝑖𝐿𝑘∥
Í𝑀
𝑘=1 𝜔𝑘
𝑖
=
Í𝑀
𝑘=1 𝜔𝑘
𝑖
√︄
𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑥
2
+

𝜕𝐿𝑘
𝜕𝝁𝑘
𝑖,𝑦
2
Í𝑀
𝑘=1 𝜔𝑘
𝑖
.
(23)
C
ADDITIONAL RESULTS
Additional qualitative experiments results. Figure 8 and Fig-
ure 9 provide more visualization results of baselines and our meth-
ods. Figure 10 provide more visualizations on depth maps of base-
lines and our methods

<!-- page 13 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
3D-GS
Abs-GS
Pixel-GS
ReAct-GS (ours)
GT
Figure 8: Qualitative comparisons of different methods on indoor scenes.

<!-- page 14 -->
MM ’25, October 27–31, 2025, Dublin, Ireland
Yuxin Cheng et al.
3D-GS
Abs-GS
Pixel-GS
ReAct-GS (ours)
GT
Figure 9: Qualitative comparisons of different methods on outdoor scenes.

<!-- page 15 -->
Re-Activating Frozen Primitives for 3D Gaussian Splatting
MM ’25, October 27–31, 2025, Dublin, Ireland
3D-GS
Abs-GS
Pixel-GS
ReAct-GS (ours)
GT
Blur-split
Figure 10: Depth visualization of different methods on various scenes.
