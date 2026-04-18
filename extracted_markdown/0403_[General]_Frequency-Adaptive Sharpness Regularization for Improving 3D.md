<!-- page 1 -->
Frequency-Adaptive Sharpness Regularization
for Improving 3D Gaussian Splatting Generalization
Youngsik Yun
Yonsei University
bbangsik@yonsei.ac.kr
Dongjun Gu
UNIST
djku1020@unist.ac.kr
Youngjung Uh
Yonsei University
yj.uh@yonsei.ac.kr
Abstract
Despite 3D Gaussian Splatting (3DGS) excelling in most
configurations, it lacks generalization across novel view-
points in a few-shot scenario because it overfits to the
sparse observations. We revisit 3DGS optimization from a
machine learning perspective, framing novel view synthe-
sis as a generalization problem to unseen viewpoints—an
underexplored direction. We propose Frequency-Adaptive
Sharpness Regularization (FASR), which reformulates the
3DGS training objective, thereby guiding 3DGS to con-
verge toward a better generalization solution.
Although
Sharpness-Aware Minimization (SAM) similarly reduces the
sharpness of the loss landscape to improve generalization
of classification models, directly employing it to 3DGS
is suboptimal due to the discrepancy between the tasks.
Specifically, it hinders reconstructing high-frequency de-
tails due to excessive regularization, while reducing its
strength leads to under-penalizing sharpness. To address
this, we reflect the local frequency of images to set the
regularization weight and the neighborhood radius when
estimating the local sharpness.
It prevents floater arti-
facts in novel viewpoints and reconstructs fine details that
SAM tends to oversmooth.
Across datasets with various
configurations, our method consistently improves a wide
range of baselines.
Code will be available at https:
//bbangsik13.github.io/FASR.
1. Introduction
Reconstructing 3D scenes from multi-view 2D images has
been a long-standing problem of interest.
3D Gaussian
Splatting (3DGS) [23] achieves photo-realistic fidelity in
novel view synthesis with real-time rendering. However,
it requires densely captured input views, which are costly to
obtain. In sparse-view settings, they often overfit to training
views, resulting in poor generalization to novel viewpoints
with unresolved details and floating artifacts. To improve
+ Ours : .00927  ± .00002
+ Ours : .03078  ± .00017
3DGS  : .00660  ± .00002
3DGS  : .03338  ± .00034
Average Error train ↓ 
Average Error test ↓ 
Figure 1. Overview. Our proposed optimization algorithm im-
proves generalization. Given eight training views rendered from
the lego scene in Blender synthetic dataset [41], our method
maintains low Average Error [44] across interpolated novel views,
whereas 3DGS exhibits overfitting. Plots are means and standard
deviations over ten runs.
generalization, previous approaches have adopted various
strategies, such as integrating geometric priors from depth
and flow estimators [8, 66], and leveraging dense corre-
spondence prediction models [20]. Although these methods
show effectiveness, 3DGS optimization has been underex-
plored as a machine learning problem where generalization
to novel viewpoints is important.
In this paper, we propose an optimization algorithm for
3DGS variants to improve quality in novel viewpoints, i.e.,
generalization (Fig. 1). Our motivation is to optimize the
model toward the flat minima in the loss landscape, which
are widely known to promote better generalization. As il-
lustrated in Fig. 2, sharp minima1 exhibit a large gap be-
tween train and test loss, whereas flat minima show a much
1We omit “local” for brevity.
arXiv:2511.17918v1  [cs.CV]  22 Nov 2025

<!-- page 2 -->
smaller gap, indicating better generalization. For example,
floaters are invisible in training views but critically emerge
in novel viewpoints; i.e., the Gaussians overfit to the train-
ing views. This large generalization gap implies a sharp
minimum, i.e., loss dramatically changes even with slight
perturbations on the model parameters2.
Although Sharpness-Aware Minimization (SAM) [12]
theoretically and empirically demonstrates that pursuing
flatter minima leads to improved generalization of classi-
fication networks, achieving flat minima does not always
guarantee better generalization in reconstruction tasks be-
cause the importance of sharpness for accurate reconstruc-
tion varies across regions. In particular, model parameters
representing high-frequency details (e.g., edges) inherently
induce a sharp loss landscape; even small changes in a well-
fitted Gaussian cause a drastic change in the loss. In con-
trast, for parameters representing low-frequency regions,
the loss changes more gradually, favoring flat minima. In
this sense, finding a flat minimum for all Gaussians is prob-
lematic; sharpness in high-frequency areas is desirable for
accurate reconstruction, flatter minima in low-frequency re-
gions are preferable for better generalization.
To this end, we propose Frequency-Adaptive Sharpness
Regularization (FASR), an optimization algorithm that pe-
nalizes local sharpness, where both the neighborhood radius
used for its computation and the regularization weight are
set in inverse proportion to the local frequency, encourag-
ing a flatter loss landscape while retaining the sharpness
required for fine details. We show that the enhancement
of our method is complementary to prior methods, thereby
providing additional performance gains across them.
Our contributions are significant as follows:
• To the best of our knowledge, we present the first fun-
damental investigation into the relationship between loss
landscape and generalization in novel view synthesis.
• Consequently, we propose an optimization algorithm for
3DGS by reformulating SAM in a frequency-adaptive
manner, overcoming the limitation of SAM, which over-
smooths high-frequency details.
• Our method is versatile in improving a wide range of
baseline methods on various datasets.
2. Related work
2.1. Sparse-view reconstruction
Reconstructing scenes from highly sparse inputs remains a
fundamental challenge, as limited supervision often leads
2Camera perturbations can be interpreted as Gaussian extrinsic param-
eter perturbations, e.g., parallel movement of all Gaussians is equivalent to
moving the camera.
3Figure is reproduced from Keskar et al. [24]. As empirically shown
in Izmailov et al. [18], the test loss landscape tends to shift relative to the
train loss landscape, while Liu et al. [37] showed that the loss landscape
differs with the imbalance level of the dataset.
Flat Minimum
Sharp Minimum
Train Loss
Test Loss
Small Generalization Gap
Large Generalization Gap
Parameters
Loss
Figure 2. Conceptual 1D Loss Landscape of Flat and Sharp
Minima3. Flat minimum better generalize then sharp minimum.
models to overfit training views, resulting in insufficient
generalization in unseen views. Early efforts sought to ex-
tend neural radiance fields (NeRFs) [41] by incorporating
additional regularizations or auxiliary cues. For example,
these studies [9, 44, 54, 60] introduce geometry- or depth-
based constraints and frequency regularization to alleviate
the underconstrained nature of sparse-view optimization.
Although these approaches demonstrate effectiveness, they
inherit the slow training and rendering processes of NeRF.
Recent studies have shifted to use 3D Gaussian Splat-
ting (3DGS) [23], aiming for real-time rendering efficiency.
These studies [8, 20, 31, 58, 66, 68] similary leverage ex-
ternal priors such as depth [6, 46], correspondence [30],
or flow [49], while another focuses on ensemble-like reg-
ularization [45, 63, 65].
Sharing a key insight with our
method, Sparfels [21] aims to minimize a worst-case loss
for robustness; however, it does so by freezing the Gaus-
sian means and approximating the objective with an upper
bound, which simplifies to a color variance regularization
along each ray. Although this proxy effectively improves
details, it may overlook color-matched floaters. In contrast,
our method directly optimizes the worst-case loss with re-
spect to all Gaussian parameters, interpreting it as a sharp-
ness regularization.
Importantly, our optimization algorithm is complemen-
tary: rather than altering the representation or adding priors,
we fundamentally guide 3DGS to converge on a general so-
lution in under-constrained sparse supervision.
2.2. Flatness and generalization
The relationship between the flatness of the loss landscape
and model generalization has been extensively investigated
in prior research. Keskar et al. [24] shows that converging
to sharp minima leads to poor generalization, while Jiang
et al. [22] identifies that sharpness is the most correlated in-
dicator of generalization. Subsequent work [7, 18] demon-
strated that averaging parameters along training trajectories
can lead to flatter minima with better generalization.
Beyond averaging strategies, Sharpness-Aware Mini-
mization (SAM) [12] explicitly estimates and reduces
sharpness, inspiring subsequent work [2, 34, 42, 51] to an-
alyze and improve upon its formulation. Moreover, some
works have analyzed the accuracy of estimated sharpness
as a measure of generalization. They show that sharpness

<!-- page 3 -->
changes with parameter rescalings [10], and the appropri-
ate sharpness estimation differs across different training se-
tups and tasks [3], which hinders the correlation between
sharpness and generalization. Consequently, recent work
introduces normalization and invariant formulations to ap-
propriately estimate sharpness, enabling a more reliable link
between sharpness and generalization [16, 19, 26, 28, 53].
On the other hand, some works report counterexamples
where sharper models generalize well, indicating that flatter
minima are not always the optimal strategy for generaliza-
tion [4, 10, 56].
Building on this perspective, we revisit the loss land-
scape sharpness in the context of the reconstruction task.
We hypothesize that the optimal sharpness is not uniform
but varies with the local frequency of the signal. Further-
more, the radius for estimating the local sharpness should
vary with the local frequency. Therefore, instead of pursu-
ing a universally flat minimum, we introduce a frequency-
adaptive optimization strategy that appropriately regularizes
sharpness for the reconstruction task, allowing sharp min-
ima where they benefit high-frequency details.
2.3. Reconstructing with perturbation
Random perturbation has been introduced into radiance
field training for two primary purposes.
Unlike our approach, which aims to improve general-
ization, some studies employ perturbation for uncertainty
quantification rather than relying on a deterministic ap-
proach. Stochastic or Bayesian formulations of NeRF have
explicitly modeled radiance or density distributions [29,
48]. Subsequent work perturbs trained models to estimate
epistemic uncertainty [15].
Similar ideas have been ex-
tended to 3DGS, where Gaussian parameters are sampled
from learned distributions to render both images and cali-
brated uncertainty maps [1].
Some works [13, 25, 36] achieve robustness by inject-
ing random noise into Gaussian parameters or query loca-
tions, which can be interpreted as randomly choosing pa-
rameters in a neighborhood radius. Nevertheless, this ran-
dom perturbation serves as an inefficient proxy for finding
the worst-case loss and may also introduce unexpected arti-
facts (Sec. C provides more details). In contrast, our method
adversarially perturbs parameters along the gradient direc-
tion to calculate the worst-case loss within that radius.
3. Method
We first provide preliminary on SAM [12] (Sec. 3.1). Then,
we explore applying SAM to 3DGS (Sec. 3.2).
Finally,
Sec. 3.3 presents our frequency-adaptive sharpness regular-
ization.
3.1. Preliminary: Sharpness-Aware Minimization
Sharpness-Aware Minimization (SAM) [12] is an optimiza-
tion method that improves generalization by encouraging
solutions to lie in flat regions of the loss landscape. Along
with minimizing the empirical loss at a parameter point w,
SAM estimates the loss sharpness within a neighborhood of
radius ρ. The optimization then jointly minimizes both the
empirical loss and this estimated sharpness:
LSAM ≜
loss sharpness
z
}|
{
max
∥ϵ∥2≤ρ L(w + ϵ) −L(w) +
empirical loss
z }| {
L(w)
= max
∥ϵ∥2≤ρ L(w + ϵ)
|
{z
}
worst-case loss
,
(1)
which is equivalent to the worst-case loss. In practice, it is
approximated via first-order Taylor expansion for the loss
function around the current parameters, perturbing the pa-
rameters in the gradient direction with magnitude ρ:
ˆϵ(w) ≜arg max
∥ϵ∥2≤ρ
L(w + ϵ) ≈ρ ·
∇wL(w)
∥∇wL(w)∥2
.
(2)
Then, the model parameters w are updated via a Stochastic
Gradient Descent (SGD) [43] step, with the gradient com-
puted at the estimated local maximum w + ˆϵ(w):
w ←w −λlr · ∇wL(w)|w+ˆϵ(w),
where λlr is a learning rate.
Supporting the intuition of
Fig. 2, this method improves generalization by tightening
the generalization bound based on sharpness derived from
the PAC-Bayesian framework [11, 39, 47]:
Theorem 1. For any ρ > 0, with training set S from data
distribution D,
LD(w) ≤max
∥ϵ∥2≤ρ LS(w + ϵ) + h(∥w∥2
2/ρ2),
where h : R+ →R+ is a strictly increasing function (under
some technical conditions on LD(w)).
In practice, ρ is a hyperparameter.
The sharpness can be interpreted as a regularization
term, as in Weighted SAM (WSAM) [61]:
LWSAM ≜L(w) +
γ
1 −γ [ max
∥ϵ∥2≤ρ L(w + ϵ) −L(w)]
= 1 −2γ
1 −γ L(w) +
γ
1 −γ max
∥ϵ∥2≤ρ L(w + ϵ).
(3)
When the weight hyperparameter γ is set to 0, the sharp-
ness term vanishes and the optimization minimizes the em-
pirical loss only. When γ = 0.5, WSAM is equivalent to
SAM (Eq. (1)), while for 0.5 < γ < 1, the sharpness term

<!-- page 4 -->
Algorithm 1 Applying SAM to 3DGS
Input: Multi-view images ˜Iv, where camera v ∈V
Output: Optimized G = (µ, q, s, σ, YDC, YAC)
1: while G not converged do
2:
L ←Loss(Render(G, v), ˜Iv)
// Get loss
3:
for all θ ∈G do
4:
ˆθ ←θ + ρθ
∇θL
∥∇θL∥2
// Ascent step
5:
end for
6:
ˆG ←(ˆθ | θ ∈G)
// Local maximum
7:
ˆL ←Loss(Render( ˆG, v), ˜Iv)
// Get loss
8:
G ←G −Adam(∇ˆG ˆL)
// Descent step
9: end while
is emphasized. Accordingly, in the extended Theorem 1,
max∥ϵ∥2≤ρ LS(w + ϵ) = LSAM
S
becomes LWSAM
S
, and the
function h is modified.
See Foret et al. [12] and Yue et al. [61] for the full theo-
rem statement and proof.
3.2. Applying SAM to 3DGS
Let
Gi
be
the
i-th
Gaussian,
defined
as
Gi
=
(µi, qi, si, σi, YDC
i , YAC
i ), where µ, q, s, σ, and Y repre-
sent the mean, rotation, scale, opacity and spherical har-
monics (SH) coefficients, respectively, with YDC and YAC
corresponding to the DC and AC terms of SH.
As in the 3DGS [23] pipeline, we render 3D Gaussians
G = (Gi)N
i=1 from the camera view v and compute the loss
L (Algorithm 1.2). The gradient of this loss guides the pa-
rameter to its local maximum (Algorithm 1.3-6). Similar
to the SAM pipeline, we then evaluate the loss ˆL at the lo-
cal maximum and use this gradient to update the original
parameters via the Adam optimizer [27] (Algorithm 1.7-8).
However, this direct application often leads to subopti-
mal performance as shown in Sec. 4.2.1. Unlike the clas-
sification task, in the reconstruction task, the curvature of
the landscape is highly correlated with the image frequency.
Loss in high-frequency regions requires sharp minima for
accurate reconstruction, making it sensitive to perturba-
tions, whereas low-frequency regions require flat minima
and are less sensitive to them. Due to this differing sen-
sitivity, a fixed ρθ is problematic.
Specifically, in high-
frequency regions, it leads to inaccurate sharpness estima-
tion by causing large first-order approximation errors (Eq.
2). Conversely, in low-frequency regions, the perturbation
is too weak to be meaningful, limiting the improvement of
SAM. Moreover, SAM penalizes estimated sharpness with
the fixed regularization weight γ = 0.5. It over-penalizes
the high-frequency regions, which must remain sharp to
preserve details. Concurrently, it under-penalizes the low-
frequency regions, failing to sufficiently improve general-
ization.
Rendered 
images
GT images
Scale maps
Loss
Separate sharpness per-
Gaussian (sec 3.3.1)
Frequency-adaptive perturbation 
magnitude (sec 3.3.2) 
Rendered 
images
Loss
Frequency-adaptive
sharpness weighting 
(sec 3.3.3)
Sharpness
RGB loss
Final loss
High-frequency
Low-frequency
Slightly 
perturb
Strongly 
perturb
Perturb
Update
GT images
Scale maps
Figure 3. Overview of our proposed method.
Key hypothesis.
Based on this intuition, we hypothesize
that for each Gaussian, the optimal neighborhood radius ρθ
and regularization weight γ vary in correlation with image
frequency, which effectively tightens the WSAM-extended
version of the generalization bound in Theorem 1.
3.3. Frequency-Adaptive Sharpness Regularization
To address the limitation of SAM, we introduce Frequency-
Adaptive Sharpness Regularization (FASR). As illustrated
in Fig. 3, we first estimate the local sharpness of each Gaus-
sian attribute independently (Sec. 3.3.1). Then we adjust
the perturbation magnitude and the regularization weight of
each Gaussian attribute4 (Secs. 3.3.2 and 3.3.3) using the
precomputed scale map Γv at view v, which is inversely
proportional to the local frequency (detailed in Sec. E).
3.3.1. Separate sharpness per-Gaussian
Unlike neural networks, 3DGS is an explicit model, which
allows us to associate the local frequency of each pixel with
its corresponding Gaussian. To apply frequency adaptiv-
ity, we begin by computing gradient on the per-Gaussian
attribute θi ∈Gi separately rather than the entire parameter
set. Accordingly, Algorithm 1.4 becomes
ˆθi ←θi + ρθ
∇θiL
∥∇θiL∥2
.
(4)
This step has the advantage of estimating the local sharp-
ness of each Gaussian attribute independently. Specifically,
it mitigates the first-order approximation error of sharpness
by large gradient Gaussians.
3.3.2. Frequency-adaptive perturbation magnitude
Going a step further, we adapt the perturbation magnitude
according to the local image frequency. Specifically, for
each Gaussian, we query the value of the optimal scale map
at the projected center coordinate (x, y) on the 2D camera
plane, γi ←Γv(x, y). Since this value is obtained in the 2D
image space, we multiply the rendered depth at the same
location, di ←Dv(x, y), and divide by the focal length
4We apply this adaptivity to the mean µi, rotation qi, and scale si,
which are geometric attributes of the Gaussian.

<!-- page 5 -->
Table 1. Quantitative comparison. Our method improves baselines across the board.
Method
LLFF (3 views)
MipNeRF-360 (12 views)
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS [ACM ToG’23]
19.810 ± .339
.6790 ± .0078
.2145 ± .0065
18.903 ± .179
.5499 ± .0036
.3734 ± .0042
+ Ours
20.783 ± .300
.7197 ± .0032
.1965 ± .0034
19.303 ± .185
.5622 ± .0051
.3552 ± .0047
CoR-GS [ECCV’24]
20.185 ± .142
.7015 ± .0040
.2029 ± .0035
19.515 ± .243
.5733 ± .0049
.3741 ± .0066
+ Ours
20.862 ± .154
.7283 ± .0042
.1932 ± .0030
19.805 ± .233
.5833 ± .0058
.3681 ± .0069
DropGaussian [CVPR’25]
20.461 ± .212
.7070 ± .0045
.2064 ± .0047
19.514 ± .199
.5722 ± .0042
.3657 ± .0036
+ Ours
20.853 ± .227
.7295 ± .0045
.1969 ± .0040
19.625 ± .254
.5750 ± .0053
.3627 ± .0051
NexusGS [CVPR’25]
21.048 ± .049
.7382 ± .0008
.1776 ± .0009
18.506 ± .098
.5222 ± .0031
.3587 ± .0021
+ Ours
21.348 ± .078
.7511 ± .0012
.1714 ± .0011
18.736 ± .103
.5316 ± .0034
.3522 ± .0024
SE-GS [ICCV’25]
20.725 ± .217
.7203 ± .0049
.1861 ± .0058
19.931 ± .288
.5930 ± .0063
.3702 ± .0054
+ Ours
21.141 ± .223
.7403 ± .0042
.1803 ± .0038
20.135 ± .232
.5960 ± .0059
.3644 ± .0052
f to adjust it for the Gaussian mean µi and scale si, so
that Gaussians located farther from the camera v are per-
turbed more strongly in 3D space. Then, the resulting value
is multiplied by a predefined neighborhood radius to deter-
mine the final perturbation. Therefore, instead of Eq. (4)
Algorithm 1.4 becomes
ˆθi ←θi + γi
di
f ρθ
∇θiL
∥∇θiL∥2
.
(5)
As a result, the perturbation magnitude is adaptive to the
frequency of the corresponding Gaussian.
This adapta-
tion mitigates the first-order approximation error of sharp-
ness at Gaussians in high-frequency regions and perturbs
strongly in low-frequency regions. Consequently, estima-
tion of sharpness is more appropriate, thus improving gen-
eralization.
3.3.3. Frequency-adaptive sharpness weighting
Finally, we adapt the regularization weight according to the
frequency of each Gaussian, where the frequency is defined
the same as in Sec. 3.3.2. From Eq. (3), the final gradient is
a weighted combination of the gradient at the original point
and the gradient at the perturbed point:
∇ˆθi ˆL ←1 −2¯γi
1 −¯γi
∇θiL +
¯γi
1 −¯γi
∇ˆθi ˆL.
(6)
Here, ¯γi ←0.95γi/γmax, where γmax is the maximum can-
didate scale of the LoG kernel, and 0.95 is determined em-
pirically. We then update the parameter at the original point
using this weighted gradient (Algorithm 1.8).
As a result, the sharpness penalty is reduced in high-
frequency regions, preserving fine details, while applying a
stronger penalty in low-frequency regions to facilitate better
regularization.
4. Experiments
Dataset.
We evaluate our method on LLFF [40] and
MipNeRF-360 [5], following previous works [45, 63, 66,
68], where the input resolution is 8× downsampled, and 3
and 12 input views are split for LLFF and MipNeRF-360,
respectively.
Implementation.
We
choose
the
publicly
available
3DGS [23] and its follow-up works as baselines. Specifi-
cally, we choose the state-of-the-art NexusGS [66], which
leverages foundation models, and CoR-GS [63], DropGaus-
sian [45], and SE-GS [65], which do not.
Metrics.
We use PSNR, SSIM [55], and LPIPS [64]
as evaluation metrics, where LPIPS is computed with a
VGG network [50]. Additionally, we use Average Error
(AVGE) [44], which is the geometric mean of PSNR, SSIM,
and LPIPS. Considering the randomness of 3DGS, we con-
duct ten runs on the LLFF dataset and five runs on the
MipNeRF-360 dataset.
4.1. Reconstruction quality
As shown in Tab. 1, our method achieves clear gains in
all metrics, datasets, and baselines. Importantly, these im-
provements are achieved without any architectural changes
or additional priors, but only with our optimization strat-
egy. Fig. 4 show visual comparisons. Applying our method
consistently improves the baselines by correcting geometric
inaccuracies and reducing floating artifacts in novel view-
points.
These results reveal that our proposed optimiza-
tion algorithm can be seamlessly integrated into current and
future 3DGS-based frameworks, providing complementary
enhancements.

<!-- page 6 -->
Figure 4. Qualitative comparison. Please zoom on the insets in red boxes to compare reconstruction quality.
4.2. Analysis
4.2.1. Ablation study
Directly applying SAM [12] to 3DGS leads to degraded per-
formance. This naive approach strongly perturbs Gaussians
with large gradients, resulting in blurry reconstructions
(Fig. 5, second column). Moreover, it increases the first-
order approximation error of sharpness estimation, hinder-
ing the optimization process of SAM (Tab. 2, fourth row).
Separate sharpness per-Gaussian, handles each Gaussian
separately. Thus, it mitigates this issue, leading to improve-
ments in some metrics (Tab. 2, fifth row), but the results
remain blurry (Fig. 5, third column). Frequency-adaptive
sharpness weighting preserves sharpness in high-frequency
details. Meanwhile, frequency-adaptive perturbation mag-
nitude enables more faithful estimation by adaptively ad-
justing perturbation magnitude.
Applying either compo-
nent individually shows performance gains (Tab. 2, sixth
and seventh rows). However, removing either component
degrades performance (Fig. 5, fourth and fifth columns).
Using both achieves a better generalization, balancing be-
tween sharpness reduction and detail preservation (Fig. 5,
sixth column; Tab. 2, eighth row).
4.2.2. Loss landscape visualization
To analyze the convergence behavior of our method, we
visualize the reconstruction loss landscape and the corre-
sponding optimization trajectories. We first train 3DGS for
5k iterations.
Subsequently, we continue training for an
additional 5k iterations with the densification process dis-
abled, under three distinct settings: 3DGS optimization,
SAM, and our proposed method.
For visualization, we
project the high-dimensional parameter trajectories onto a
2D plane using Principal Component Analysis with param-
eters from both trajectories.

<!-- page 7 -->
Figure 5. Ablation study. FAP and FAS denote frequency-adaptive perturbation magnitude and frequency-adaptive sharpness weighting,
respectively. “3DGS”, “SAM”, and “w/o FAS & FAP” produce inaccurate geometry (red box). All except “Ours Full” show blurry results
(yellow box).
Table 2. Ablation study. We report averaged results over ten
runs. Standard deviations are omitted due to space constraints.
SSG, FAP, and FAS denote separate sharpness per-Gaussian, fre-
quency adaptive perturbation magnitude, and frequency adaptive
sharpness weighting, respectively.
Components
LLFF (3 views)
SAM
SSG
FAP
FAS
PSNR ↑
SSIM ↑
LPIPS ↓
✗
✗
✗
✗
19.810
.6790
.2145
✓
✗
✗
✗
20.198
.6958
.2095
✓
✓
✗
✗
20.390
.6977
.2174
✓
✓
✗
✓
20.570
.6980
.2142
✓
✓
✓
✗
20.560
.7116
.2023
✓
✓
✓
✓
20.783
.7197
.1965
SAM converges to flatter minima than 3DGS. In Fig. 6a,
the loss range between the local maximum and minimum is
2.37×, and the measured sharpness λmax is 1.49× smaller.
SAM also achieves a test loss of 0.0129 lower than 3DGS,
narrowing the generalization gap from 0.0985 to 0.0809.
However, our method converges to less flat minima than
SAM. As shown in Fig. 6b, the local loss range and mea-
sured sharpness of ours are 1.11× smaller and 1.01×
smaller than 3DGS, respectively. Interestingly, our method
achieves test loss of 0.0141 lower than 3DGS and further
narrows the generalization gap to 0.0805, outperforming
SAM in generalization.
These results suggest that sharpness is not strictly cor-
related with generalization, supporting our hypothesis that
sharpness of high-frequency pixel should be preserved.
This aligns with our finding (Fig. 5, second column) that
SAM tends to over-penalize high-frequency details, leading
to blurry results.
4.2.3. Performance improvement by covisibility level
The improvement from our method is more substantial
in regions observed by fewer training views.
Follow-
ing CoMapGS [20], we compute covisibility maps using
MASt3R [30]. As shown in Tab. 3, the improvement of
our method over the baseline 3DGS increases progressively
as the covisibility decreases. This behavior aligns with our
Range = 7.1e-5
Range = 6.4e-5
λmax = 2.81e-3
Range = 7.1e-5
λmax = 2.81e-3
λmax = 2.77e-3
Range = 3.0e-5
λmax = 1.88e-3
0.0083
0.0888
0.0091
0.0900
(a) 3DGS vs. SAM
(b) 3DGS vs. Ours
0.0044
0.0044
0.1029
0.1029
Figure 6. Loss landscape visualization. We compare the conver-
gence behaviors of 3DGS, SAM, and Ours. Because the visualiza-
tion produces a smoothed loss landscape, we provide a zoomed-in
view near the convergence points. We measure sharpness as the
maximum eigenvalue λmax of the Hessian matrix [38, 57].
Table 3. Performance by covisibility level on LLFF dataset.
Our method shows greater improvement with higher view sparsity.
Covisibility level
AVGE ↓
3DGS
+ Ours
∆
Covisibility 3
.0483 ±.0093
.0417 ±.0080
- .0066 ±.0043
Covisibility 2
.0757 ±.0156
.0644 ±.0137
- .0114 ±.0067
Covisibility 1
.0948 ±.0256
.0806 ±.0232
- .0142 ±.0074
hypothesis that our method enhances generalization, partic-
ularly in under-constrained regions.
4.3. Application
4.3.1. Reducing computation cost of FASR
The main limitation of SAM and its follow-up work is that
they compute the loss gradient twice at each step, theo-
retically doubling the training time. Fortunately, applying
SAM on the last few training epochs improves performance
similarly to the full application [67]. Based on this find-
ing, we apply our method during the last 12.5% of the total
iterations, denoted as Ours-L. As shown in Tab. 4, Ours-

<!-- page 8 -->
Table 4. Applying FASR at late training phase. For efficient
training, we apply our method during the later iterations, denoted
as Ours-L. We report metrics (and their changes relative to 3DGS) av-
eraged over ten runs on an RTX A5000; standard devia-
tions are omitted for brevity. Bold indicates the best perfor-
mance, and underline indicates the second best.
Method
LLFF (3 views)
AVGE ↓
Time (sec) ↓
3DGS
.1111
85.7
3DGS + Ours
.0979 (-.0132)
237. (2.77×)
3DGS + Ours-L
.1016 (-.0095)
98.2 (1.15×)
Table 5. Applying FASR to online dynamic 3D Gaussians. Our
method is effective in dynamic scenarios under temporal sparsity,
notably improving temporal consistency by reducing mTV.
Method
Neural 3D Video
PSNR ↑
SSIM ↑
mTV ↓
Yun et al. [SIGGRAPH’25]
32.542
.9486
.1109
+ Ours
32.622
.9497
.0989
L improves 3DGS by 0.0095 in AVGE, which are slightly
(by 0.0037) less than full Ours. Moreover, compared to the
baseline 3DGS, Ours increases the training time by 2.77×,
Ours-L increases it by only 1.15×, making it more efficient.
The performance degradation of Ours-L occurs because
3DGS adaptively adjusts its number of learnable param-
eters through densification—a key difference from neural
networks. This makes early training iterations influential,
leading different results from Zhou et al. [67].
4.3.2. Improving generalization in temporal sparsity
Additionally, we extend our method to dynamic scenes
captured with multi-view cameras, the Neural 3D Video
dataset [33]. Specifically, we conduct experiments in an on-
line configuration [14, 17, 32, 59], where observations are
spatially dense but temporally sparse, meaning that we can
only access the current frame in a sequentially processed
video stream. We applied our method to Yun et al. [62]
with 3DGStream [52] backbone.
Following their proto-
col, we select the first frame 3D Gaussians with the high-
est PSNR for initialization and compute the masked total
variation (mTV) to measure temporal consistency.
Our method improves temporal consistency and visual
quality compared to the baseline (Tab. 5). This shows that
our approach enhances generalization not only in the spatial
domain but also in the temporal domain. Furthermore, Yun
et al. [62] claims that one cause of temporal jittering is the
inevitable noise in training datasets. Since SAM shows ro-
bustness on noisy training data, our finding aligns with this
explanation.
Figure 7. Qualitative comparison with FreeNeRF. Please zoom
on the insets in red boxes to compare reconstruction quality.
Table 6.
Quantitative comparison with FreeNeRF. Our ap-
proach improves the NeRF baseline.
Method
LLFF (3 views)
PSNR ↑
SSIM ↑
LPIPS ↓
FreeNeRF [CVPR’23]
19.523
.6063
.3103
+ Ours
19.584
.6182
.2983
4.3.3. Applying our intuition to NeRF
Despite our method is not directly designed for implicit rep-
resentations such as NeRF [41] because a single parameter
affects all pixels, we can alternatively apply our intuition to
baselines that learn in a coarse-to-fine procedure [36, 60].
Specifically, we set a strong perturbation magnitude and
regularization weight when the NeRF parameters learn low-
frequency components, and gradually decrease these values
as the model learns high-frequency details.
We applied this approach to FreeNeRF [60] and demon-
strate that it improves both visual quality in novel view syn-
thesis (Fig. 7) and quantitative results (Tab. 6). Although
the improvement over the NeRF baseline is smaller than
that in the 3DGS baselines due to discrepancy in the rep-
resentation, these additional gains still support our intuition
and versatility across different representations.
5. Conclusion
In this work, we present the first fundamental investiga-
tion linking loss landscape and generalization to novel
view synthesis, thereby improving quality of 3D Gaus-
sian Splatting [23], especially in sparse view reconstruc-
tion. We propose Frequency-Adaptive Sharpness Regular-
ization (FASR), an optimization algorithm that reformulates
Sharpness-Aware Minimization (SAM) [12] in a frequency-
adaptive manner. This reformulation overcomes the limita-
tion of SAM in reconstruction tasks, achieving generaliza-
tion as well as fine detail reconstruction. FASR is easily
applicable across diverse baselines and can be further ex-
tended to NeRF-based models and dynamic scenes in tem-
porally sparse scenarios. We hope this work inspires the
research community to explore the link between sharpness
and generalization in reconstruction fields.

<!-- page 9 -->
References
[1] Luca Savant Aira, Diego Valsesia, and Enrico Magli. Mod-
eling uncertainty for gaussian splatting. IEEE Transactions
on Neural Networks and Learning Systems, 36(6):11657–
11663, 2025. 3
[2] Maksym Andriushchenko and Nicolas Flammarion. Towards
understanding sharpness-aware minimization. In Proceed-
ings of the 39th International Conference on Machine Learn-
ing, pages 639–668. PMLR, 2022. 2
[3] Maksym Andriushchenko, Francesco Croce, Maximilian
M¨uller, Matthias Hein, and Nicolas Flammarion. A modern
look at the relationship between sharpness and generaliza-
tion. In Proceedings of the 40th International Conference on
Machine Learning, pages 840–902. PMLR, 2023. 3
[4] Maksym Andriushchenko, Francesco Croce, Maximilian
M¨uller, Matthias Hein, and Nicolas Flammarion. A modern
look at the relationship between sharpness and generaliza-
tion. In Proceedings of the 40th International Conference on
Machine Learning, pages 840–902. PMLR, 2023. 3
[5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5470–5479, 2022. 5
[6] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka,
and Matthias M¨uller. Zoedepth: Zero-shot transfer by com-
bining relative and metric depth, 2023. 2
[7] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol
Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park.
Swad: Domain generalization by seeking flat minima. In
Advances in Neural Information Processing Systems, pages
22405–22418. Curran Associates, Inc., 2021. 2
[8] Dongrui Dai and Yuxiang Xing. Eap-gs: Efficient augmenta-
tion of pointcloud for 3d gaussian splatting in few-shot scene
reconstruction. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
16498–16507, 2025. 1, 2
[9] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ra-
manan. Depth-supervised nerf: Fewer views and faster train-
ing for free. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
12882–12891, 2022. 2
[10] Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua
Bengio. Sharp minima can generalize for deep nets. In Pro-
ceedings of the 34th International Conference on Machine
Learning, pages 1019–1028. PMLR, 2017. 3
[11] Gintare Karolina Dziugaite and Daniel M. Roy. Comput-
ing nonvacuous generalization bounds for deep (stochastic)
neural networks with many more parameters than training
data. In Proceedings of the 33rd Annual Conference on Un-
certainty in Artificial Intelligence (UAI), 2017. 3
[12] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam
Neyshabur.
Sharpness-aware minimization for efficiently
improving generalization.
In International Conference on
Learning Representations, 2021. 2, 3, 4, 6, 8
[13] Qiankun Gao, Jiarui Meng, Chengxiang Wen, Jie Chen, and
Jian Zhang. Hicom: Hierarchical coherent motion for dy-
namic streamable scenes with 3d gaussian splatting.
In
Advances in Neural Information Processing Systems, pages
80609–80633. Curran Associates, Inc., 2024. 3
[14] Sharath Girish, Tianye Li, Amrita Mazumdar, Abhinav Shri-
vastava, David Luebke, and Shalini De Mello. Queen: Quan-
tized efficient encoding of dynamic gaussians for streaming
free-viewpoint videos. In Advances in Neural Information
Processing Systems, pages 43435–43467. Curran Associates,
Inc., 2024. 8
[15] Lily Goli, Cody Reading, Silvia Sell´an, Alec Jacobson,
and Andrea Tagliasacchi. Bayes’ rays: Uncertainty quan-
tification for neural radiance fields.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 20061–20070, 2024. 3
[16] Moritz Haas, Jin Xu, Volkan Cevher, and Leena Chen-
nuru Vankadara. µp2: Effective sharpness aware minimiza-
tion requires layerwise perturbation scaling.
In Advances
in Neural Information Processing Systems, pages 38888–
38959. Curran Associates, Inc., 2024. 3
[17] Qiang Hu, Zihan Zheng, Houqiang Zhong, Sihua Fu, Li
Song, Xiaoyun Zhang, Guangtao Zhai, and Yanfeng Wang.
4dgc:
Rate-aware 4d gaussian compression for efficient
streamable free-viewpoint video.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 875–885, 2025. 8
[18] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry
Vetrov, and Andrew Gordon Wilson.
Averaging weights
leads to wider optima and better generalization. In Proceed-
ings of the 34th Annual Conference on Uncertainty in Artifi-
cial Intelligence (UAI), 2018. 2
[19] Cheongjae Jang, Sungyoon Lee, Frank Park, and Yung-Kyun
Noh. A reparametrization-invariant sharpness measure based
on information geometry. Advances in neural information
processing systems, 35:27893–27905, 2022. 3
[20] Youngkyoon Jang and Eduardo P´erez-Pellitero. Comapgs:
Covisibility map-based gaussian splatting for sparse novel
view synthesis.
In Proceedings of the Computer Vision
and Pattern Recognition Conference (CVPR), pages 26779–
26788, 2025. 1, 2, 7
[21] Shubhendu Jena, Amine Ouasfi, Mae Younes, and Adnane
Boukhayma.
Sparfels:
Fast reconstruction from sparse
unposed imagery.
In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pages
27476–27487, 2025. 2
[22] Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip
Krishnan, and Samy Bengio. Fantastic generalization mea-
sures and where to find them. In International Conference
on Learning Representations, 2020. 2
[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4), 2023.
1, 2, 4, 5, 8
[24] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal,
Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-
batch training for deep learning: Generalization gap and
sharp minima. In International Conference on Learning Rep-
resentations, 2017. 2

<!-- page 10 -->
[25] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. In Advances in Neural
Information Processing Systems, pages 80965–80986. Cur-
ran Associates, Inc., 2024. 3
[26] Minyoung Kim,
Da Li,
Shell X Hu,
and Timothy
Hospedales. Fisher SAM: Information geometry and sharp-
ness aware minimisation.
In Proceedings of the 39th In-
ternational Conference on Machine Learning, pages 11148–
11161. PMLR, 2022. 3
[27] Diederik P. Kingma and Jimmy Ba.
Adam: A method
for stochastic optimization. In International Conference on
Learning Representations, 2015. 4
[28] Jungmin Kwon,
Jeongseop Kim,
Hyunseo Park,
and
In Kwon Choi. Asam: Adaptive sharpness-aware minimiza-
tion for scale-invariant learning of deep neural networks. In
Proceedings of the 38th International Conference on Ma-
chine Learning, pages 5905–5914. PMLR, 2021. 3
[29] Sibaek Lee, Kyeongsu Kang, Seongbo Ha, and Hyeonwoo
Yu.
Bayesian nerf: Quantifying uncertainty with volume
density for neural implicit fields. IEEE Robotics and Au-
tomation Letters, 10(3):2144–2151, 2025. 3
[30] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Ground-
ing image matching in 3d with mast3r. In Computer Vision
– ECCV 2024, pages 71–91, Cham, 2025. Springer Nature
Switzerland. 2, 7
[31] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normaliza-
tion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 20775–
20785, 2024. 2
[32] Lingzhi LI, Zhen Shen, Zhongshu Wang, Li Shen, and Ping
Tan. Streaming radiance fields for 3d video synthesis. In
Advances in Neural Information Processing Systems, pages
13485–13498. Curran Associates, Inc., 2022. 8
[33] Tianye Li, Mira Slavcheva, Michael Zollh¨ofer, Simon Green,
Christoph Lassner, Changil Kim, Tanner Schmidt, Steven
Lovegrove, Michael Goesele, Richard Newcombe, and
Zhaoyang Lv. Neural 3d video synthesis from multi-view
video. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 5521–
5531, 2022. 8
[34] Tao Li, Pan Zhou, Zhengbao He, Xinwen Cheng, and Xiaolin
Huang. Friendly sharpness-aware minimization. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 5631–5640, 2024. 2
[35] Tony Lindeberg. Scale selection properties of generalized
scale-space interest point detectors. Journal of Mathematical
Imaging and Vision, 46(2):177–210, 2013. 13
[36] Selena Ling, Merlin Nimier-David, Alec Jacobson, and
Nicholas Sharp. Stochastic preconditioning for neural field
optimization. ACM Trans. Graph., 44(4), 2025. 3, 8
[37] Yahao Liu, Qin Wang, Lixin Duan, and Wen Li. Balanced
sharpness-aware minimization for imbalanced regression. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pages 6242–6251, 2025. 2
[38] Haocheng Luo, Tuan Truong, Tung Pham, Mehrtash Ha-
randi, Dinh Phung, and Trung Le.
Explicit eigenvalue
regularization improves sharpness-aware minimization. In
The Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. 7
[39] David A. McAllester.
Some pac-bayesian theorems.
In
Proceedings of the Eleventh Annual Conference on Com-
putational Learning Theory, page 230–234, New York, NY,
USA, 1998. Association for Computing Machinery. 3
[40] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: practical view syn-
thesis with prescriptive sampling guidelines.
ACM Trans.
Graph., 38(4), 2019. 5
[41] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In Computer Vision – ECCV 2020, pages 405–421,
Cham, 2020. Springer International Publishing. 1, 2, 8
[42] Maximilian Mueller, Tiffany Vlaar, David Rolnick, and
Matthias Hein. Normalization layers are all that sharpness-
aware minimization needs. In Advances in Neural Informa-
tion Processing Systems, pages 69228–69252. Curran Asso-
ciates, Inc., 2023. 2
[43] Y. Nesterov. A method for solving the convex programming
problem with convergence rate o(1/k2), 1983. 3
[44] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall,
Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan.
Regnerf: Regularizing neural radiance fields for view syn-
thesis from sparse inputs. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5480–5490, 2022. 1, 2, 5
[45] Hyunwoo Park, Gun Ryu, and Wonjun Kim.
Dropgaus-
sian:
Structural regularization for sparse-view gaussian
splatting. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
21600–21609, 2025. 2, 5
[46] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF International Conference on Computer Vision
(ICCV), pages 12179–12188, 2021. 2
[47] John Shawe-Taylor and Robert C. Williamson. A pac analy-
sis of a bayesian estimator. In Proceedings of the Tenth An-
nual Conference on Computational Learning Theory, page
2–9, New York, NY, USA, 1997. Association for Computing
Machinery. 3
[48] Jianxiong Shen, Adria Ruiz, Antonio Agudo, and Francesc
Moreno-Noguer. Stochastic neural radiance fields: Quanti-
fying uncertainty in implicit 3d representations. In 2021 In-
ternational Conference on 3D Vision (3DV), pages 972–981,
2021. 3
[49] Xiaoyu Shi, Zhaoyang Huang, Dasong Li, Manyuan Zhang,
Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and
Hongsheng Li. Flowformer++: Masked cost volume autoen-
coding for pretraining optical flow estimation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 1599–1610, 2023. 2, 13

<!-- page 11 -->
[50] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition. In In-
ternational Conference on Learning Representations, 2015.
5
[51] Hao Sun, Li Shen, Qihuang Zhong, Liang Ding, Shixiang
Chen, Jingwei Sun, Jing Li, Guangzhong Sun, and Dacheng
Tao. Adasam: Boosting sharpness-aware minimization with
adaptive learning rate and momentum for training deep neu-
ral networks. Neural Networks, 169:506–519, 2024. 2
[52] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei
Zhao, and Wei Xing.
3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-
viewpoint videos. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 20675–20685, 2024. 8
[53] Yusuke Tsuzuku, Issei Sato, and Masashi Sugiyama. Nor-
malized flat minima: Exploring scale invariant definition of
flat minima for neural networks using pac-bayesian analy-
sis. In International Conference on Machine Learning, pages
9636–9647. PMLR, 2020. 3
[54] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Zi-
wei Liu. Sparsenerf: Distilling depth ranking for few-shot
novel view synthesis. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pages
9065–9076, 2023. 2
[55] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 5
[56] Kaiyue Wen, Zhiyuan Li, and Tengyu Ma. Sharpness min-
imization algorithms do not only minimize sharpness to
achieve better generalization. Advances in Neural Informa-
tion Processing Systems, 36:1024–1035, 2023. 3
[57] Kaiyue Wen, Tengyu Ma, and Zhiyuan Li. How sharpness-
aware minimization minimizes sharpness?
In The
Eleventh International Conference on Learning Representa-
tions, 2023. 7
[58] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao, Li
Li, and Yulan Guo. Dropoutgs: Dropping out gaussians for
better sparse-view rendering. In Proceedings of the Com-
puter Vision and Pattern Recognition Conference (CVPR),
pages 701–710, 2025. 2
[59] Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu
Yang, Jie Liang, Jiahao Wu, and Ronggang Wang. Instant
gaussian stream: Fast and generalizable streaming of dy-
namic scene reconstruction via gaussian splatting. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 16520–16531, 2025.
8
[60] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Im-
proving few-shot neural rendering with free frequency reg-
ularization.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
8254–8263, 2023. 2, 8
[61] Yun Yue, Jiadi Jiang, Zhiling Ye, Ning Gao, Yongchao Liu,
and Ke Zhang.
Sharpness-aware minimization revisited:
Weighted sharpness as a regularization term. In Proceedings
of the 29th ACM SIGKDD Conference on Knowledge Dis-
covery and Data Mining, page 3185–3194, New York, NY,
USA, 2023. Association for Computing Machinery. 3, 4
[62] Youngsik Yun, Jeongmin Bae, Hyunseung Son, Seoha Kim,
Hahyun Lee, Gun Bang, and Youngjung Uh. Compensat-
ing spatiotemporally inconsistent observations for online dy-
namic 3d gaussian splatting. In Proceedings of the Special
Interest Group on Computer Graphics and Interactive Tech-
niques Conference Conference Papers, New York, NY, USA,
2025. Association for Computing Machinery. 8, 13
[63] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin
Zheng, and Xiao Bai. Cor-gs: Sparse-view 3d gaussian splat-
ting via co-regularization. In Computer Vision – ECCV 2024,
pages 335–352, Cham, 2025. Springer Nature Switzerland.
2, 5
[64] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2018. 5
[65] Chen Zhao, Xuan Wang, Tong Zhang, Saqib Javed, and
Mathieu Salzmann. Self-ensembling gaussian splatting for
few-shot novel view synthesis.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision
(ICCV), pages 4940–4950, 2025. 2, 5
[66] Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun,
Junyu Dong, Huaidong Zhang, and Yong Du.
Nexusgs:
Sparse view synthesis with epipolar depth priors in 3d gaus-
sian splatting.
In Proceedings of the Computer Vision
and Pattern Recognition Conference (CVPR), pages 26800–
26809, 2025. 1, 2, 5
[67] Zhanpeng Zhou, Mingze Wang, Yuchen Mao, Bingrui Li,
and Junchi Yan. Sharpness-aware minimization efficiently
selects flatter minima late in training. In The Thirteenth In-
ternational Conference on Learning Representations, 2025.
7, 8, 13
[68] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang.
Fsgs:
Real-time few-shot view synthesis using gaussian
splatting. In Computer Vision – ECCV 2024, pages 145–163,
Cham, 2025. Springer Nature Switzerland. 2, 5

<!-- page 12 -->
2e-2
5e-2
1e-1
2e-1
5e-1
1e0
Perturbation magnitude
0.10
0.12
0.14
AVGE 
mean
Mean
± Std
Best
1e-4
2e-4
5e-4
1e-3
2e-3
5e-3
Perturbation magnitude
0.108
0.110
0.112
AVGE 
rotation
Mean
± Std
Best
5e-2
1e-1
2e-1
5e-1
1e0
2e0
5e0
Perturbation magnitude
0.110
0.115
AVGE 
scale
Mean
± Std
Best
2e-6
5e-6
1e-5
2e-5
5e-5
1e-4
Perturbation magnitude
0.108
0.110
0.112
AVGE 
opacity
Mean
± Std
Best
5e-3
1e-2
2e-2
5e-2
1e-1
2e-1
Perturbation magnitude
0.1075
0.1100
0.1125
AVGE 
SHDC
Mean
± Std
Best
2e-4
5e-4
1e-3
2e-3
5e-3
1e-2
Perturbation magnitude
0.108
0.110
0.112
AVGE 
SHAC
Mean
± Std
Best
Figure 8. Hyperparameter grid search, where each parameter is perturbed individually on the LLFF dataset. Plots are means and
standard deviations over ten runs. We mark a star at the best hyperparameter value.
A. Hyperparameter search
Selecting the optimal neighborhood of radius ρ, i.e., the
perturbation magnitude, remains a challenging problem in
SAM-based methods. Although our method scales ρ for
each Gaussian considering local image frequency, ρ itself
remains a core hyperparameter. Similar to SAM, we first
grid search the optimal perturbation magnitude by perturb-
ing each parameter individually, as shown in Fig. 8. How-
ever, the optimal ρ when perturbing all parameters simul-
taneously is typically smaller than the values found in this
search because the parameters are not independent and mu-
tually influence each other. Therefore, we select candidates
including the individually found ρ and smaller values, and
then perform a grid search with simultaneous perturbation
to determine the final hyperparameter.
B. Contribution of each Gaussian arrtibute
To analyze the contribution of each Gaussian attribute to
the overall performance, we apply our method either exclu-
sively to a single attribute or to all attributes except one.
Nevertheless, all attributes contribute positively to the per-
formance, the Gaussian mean is the dominant contributor
to the performance gain (Tab. 7). This finding suggests that
instead of perturbing all attributes simultaneously, applying
our method exclusively to the Gaussian mean could be a
simplified approach, significantly reducing hyperparameter
search complexity while likely retaining a substantial por-
tion of the performance improvement.
C. Comparison to random perturbation
To support our claims in Sec. 2.3, we demonstrate the im-
pact of applying random perturbations to Gaussian parame-
ters during training. As shown in Fig. 9, random perturba-
Table 7. Ablation study on the contribution of each Gaussian
attributes. The Gaussian mean is the dominant contributor to the
performance improvement.
Method
LLFF (3 views)
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS + Ours
20.783 ± .300
.7197 ± .0032
.1965 ± .0034
w/o mean
20.053 ± .228
.6903 ± .0051
.2102 ± .0040
w/o rotation
20.656 ± .241
.7175 ± .0046
.1976 ± .0040
w/o scale
20.641 ± .235
.7171 ± .0049
.1987 ± .0035
w/o opacity
20.628 ± .218
.7175 ± .0044
.1977 ± .0033
w/o SHDC
20.594 ± .242
.7152 ± .0052
.1972 ± .0036
w/o SHAC
20.599 ± .205
.7169 ± .0041
.1973 ± .0034
w/ mean
20.561 ± .268
.7147 ± .0046
.1975 ± .0037
w/ rotation
19.854 ± .168
.6828 ± .0029
.2110 ± .0024
w/ scale
20.036 ± .222
.6871 ± .0048
.2093 ± .0037
w/ opacity
19.888 ± .188
.6803 ± .0045
.2124 ± .0033
w/ SHDC
19.990 ± .202
.6867 ± .0048
.2115 ± .0034
w/ SHAC
19.914 ± .186
.6827 ± .0042
.2114 ± .0033
3DGS
19.810 ± .339
.6790 ± .0078
.2145 ± .0065
tions often introduce unexpected artifacts and yield smaller
performance gains than ours (Tab. 8).
Furthermore, to demonstrate the importance of finding
a local maximum by adversarial perturbation, we compare
our method with a variant that randomly samples parame-
ters within the neighborhood radius. As shown in Tab. 8, the
improvement of random perturbation is smaller than ours,
indicating that adversarially perturbing in the gradient di-
rection is essential for effective sharpness regularization.

<!-- page 13 -->
Figure 9. Qualitative comparison of random perturbation and
our method. Random perturbation often introduce unexpected
artifacts.
Method
LLFF (3 views)
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS
19.810 ± .339
.6790 ± .0078
.2145 ± .0065
w/ RP
20.174 ± .207
.6987 ± .0040
.2009 ± .0032
w/o AP
20.263 ± .289
.6990 ± .0031
.2056 ± .0027
3DGS + Ours
20.783 ± .300
.7197 ± .0032
.1965 ± .0034
Table 8. Quantitative comparison of random perturbation and
adversarial perturbation. RP and AP denote random perturba-
tion on Gaussian parameters and adversarial perturbation, respec-
tively
D. Analysis of late-phase application
We provide a detailed study on the application timing of our
method. In Sec. 4.3.1, we follow Zhou et al. [67], applying
our method only during the final 12.5% of the total training
iterations. Extending this, we further analyze the impact
of the starting point by varying the application duration in
10% increments of the total iterations. As shown in Fig. 10,
fully applying our method yields the best performance, and
the performance degrades as the application phase is later
in the training process.
E. LoG for local image frequency estimation.
We calculate the local frequency at each pixel of the input
image employing a multi-scale analysis inspired by blob de-
tection using the Laplacian of Gaussian (LoG) [35]. For the
grayscale image, we calculate the absolute LoG response at
each candidate scale. We capture the optimal scale at each
pixel by finding the first significant increase in its response
curve that exceeds the threshold. The optimal scale is the
smallest one that precedes such a rise; otherwise, we as-
sign the maximum candidate scale. This process results in
an optimal scale map Γv ∈RH×W at view v, with H and
W denoting the height and width of the image, respectively.
The local frequency is inversely proportional to this selected
0
2000
4000
6000
8000
10000
Start Iteration
0.0950
0.0975
0.1000
0.1025
0.1050
0.1075
0.1100
AVGE 
Mean
± Std
Best
Figure 10. Ablation study on the application phase. We demon-
strate the performance improvements obtained by varying the ap-
plication duration in 10% increments of the total training itera-
tions.
scale.
F. Implementation detail
For all methods except 3DGS, we use the official repos-
itory.
Since the official 3DGS does not target sparse-
view reconstruction, we refactor it by referring to CoR-
GS and DropGaussian. NexusGS does not provide optical
flow for MipNeRF-360; we compute the flow using Flow-
Former++ [49]. When the baseline utilizes regularization,
we backpropagate it after computing the gradient of FASR.
When applying our method in the deformation process, we
compute the FASR gradient on the deformed Gaussians and
propagate it back to the deformation network. We update
the residual map of Yun et al. [62] during the ascent step
and the Gaussians during the descent step. For each dataset,
we use the same values of ρθ and γ for all baselines except
NexusGS, where we set them to 0.1× the values used for
other baselines due to its use of denser per-pixel Gaussians.
G. More results
We report the quantitative results of each scene in Tab. 9.
Our method method outperforms the baseline in most cases.

<!-- page 14 -->
Table 9. Per-scene quantitative results on LLFF and MipNeRF-360 dataset. We report AVGE (lower is better) of each scene.
Method
LLFF (3 views)
fern
flower
fortress
horns
leaves
orchids
room
trex
3DGS
.0950 ±.01790
.1147 ±.00180
.0885 ±.00390
.1276 ±.00240
.1362 ±.00170
.1769 ±.00190
.0753 ±.00160
.0749 ±.00210
+ Ours
.0765 ±.00080
.1066 ±.00140
.0781 ±.00530
.1044 ±.00280
.1170 ±.00130
.1605 ±.00190
.0686 ±.00190
.0712 ±.00710
CoR-GS
.0806 ±.00080
.1088 ±.00290
.0754 ±.00160
.1205 ±.00120
.1427 ±.00310
.1735 ±.00200
.0723 ±.00130
.0699 ±.00440
+ Ours
.0724 ±.00090
.1014 ±.00220
.0683 ±.00190
.1075 ±.00200
.1326 ±.00240
.1603 ±.00100
.0695 ±.00100
.0614 ±.00340
DropGaussian
.0791 ±.00120
.1056 ±.00210
.0796 ±.00420
.1186 ±.00270
.1284 ±.00190
.1650 ±.00160
.0715 ±.00150
.0687 ±.00190
+ Ours
.0717 ±.00100
.1048 ±.00170
.0784 ±.00340
.1101 ±.00340
.1176 ±.00140
.1536 ±.00170
.0701 ±.00200
.0657 ±.00260
NexusGS
.0893 ±.00030
.0981 ±.00020
.0498 ±.00030
.0909 ±.00050
.1072 ±.00030
.1383 ±.00060
.0755 ±.00060
.0810 ±.00050
+ Ours
.0848 ±.00050
.0989 ±.00060
.0475 ±.00090
.0897 ±.00070
.0993 ±.00060
.1354 ±.00040
.0696 ±.00030
.0741 ±.00060
SE-GS
.0720 ±.00040
.1052 ±.00180
.0662 ±.00130
.1104 ±.00180
.1392 ±.00230
.1707 ±.01440
.0635 ±.00220
.0596 ±.00180
+ Ours
.0675 ±.00060
.0992 ±.00160
.0661 ±.00820
.1028 ±.00160
.1310 ±.00160
.1546 ±.00170
.0629 ±.00120
.0564 ±.00080
Method
MipNeRF-360 (12 views)
bicycle
bonsai
counter
garden
kitchen
room
stump
3DGS
.1763 ±.00400
.1408 ±.00400
.1464 ±.00250
.1302 ±.00190
.1052 ±.00160
.1078 ±.00110
.2347 ±.00280
+ Ours
.1714 ±.00120
.1285 ±.00240
.1350 ±.00110
.1240 ±.00130
.1017 ±.00160
.1021 ±.00460
.2253 ±.00640
CoR-GS
.1734 ±.00340
.1262 ±.00250
.1310 ±.00120
.1297 ±.00370
.1054 ±.00360
.0931 ±.00230
.2214 ±.00770
+ Ours
.1733 ±.00170
.1131 ±.00230
.1229 ±.00060
.1287 ±.00290
.1044 ±.00320
.0993 ±.00440
.2073 ±.00950
DropGaussian
.1646 ±.00110
.1301 ±.00210
.1356 ±.00140
.1243 ±.00110
.0979 ±.00270
.1026 ±.00500
.2187 ±.00410
+ Ours
.1613 ±.00310
.1286 ±.00070
.1309 ±.00100
.1218 ±.00200
.1027 ±.00210
.1003 ±.00440
.2171 ±.01180
NexusGS
.1818 ±.00100
.1308 ±.00190
.1563 ±.00130
.1249 ±.00030
.0941 ±.00070
.1506 ±.00380
.2352 ±.00240
+ Ours
.1739 ±.00100
.1278 ±.00060
.1531 ±.00160
.1243 ±.00040
.0921 ±.00090
.1421 ±.00450
.2299 ±.00210
SE-GS
.1624 ±.00350
.1179 ±.00300
.1251 ±.00370
.1207 ±.00160
.1107 ±.00370
.0875 ±.00380
.2014 ±.00830
+ Ours
.1599 ±.00690
.1105 ±.00300
.1217 ±.00170
.1179 ±.00250
.1092 ±.00350
.0973 ±.00090
.1941 ±.00480
