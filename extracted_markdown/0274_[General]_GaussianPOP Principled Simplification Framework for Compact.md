<!-- page 1 -->
GaussianPOP: Principled Simplification Framework for
Compact 3D Gaussian Splatting via Error Quantification
Soonbin Lee1
Yeong-Gyu Kim2
Simon Sasse1
Tom´as M. Borges1
Yago S´anchez1
Eun-Seok Ryu2
Thomas Schierl1
Cornelius Hellge1
1Fraunhofer Heinrich-Hertz-Institute (HHI), Germany
2Sungkyunkwan University (SKKU), South Korea
{first name}.{last name}@hhi.fraunhofer.de, {kyk2798, esryu}@skku.edu
①On-training time pruning
0.24M Gaussians
PSNR: 31.29dB
0.21M Gaussians
PSNR: 31.63dB
GaussianSpa
Ours
②Post-training time pruning
80% Pruned+5K Fine-tuning
PSNR: 30.77dB
LightGaussian
Ours
80% Pruned+5K Fine-tuning
PSNR: 32.05dB
Figure 1. We propose GaussianPOP, enabling high-quality and compact view synthesis with superior rendering of details. The figure
compares our method against state-of-the-art approaches in two key scenarios. The on-training scenario integrates pruning into the training
process from scratch, whereas the post-training scenario applies simplification to a pre-trained model.
Abstract
Existing 3D Gaussian Splatting simplification methods
commonly use importance scores, such as blending weights
or sensitivity, to identify redundant Gaussians. However,
these scores are not driven by visual error metrics, often
leading to suboptimal trade-offs between compactness and
rendering fidelity. We present GaussianPOP, a principled
simplification framework based on analytical Gaussian er-
ror quantification. Our key contribution is a novel error cri-
terion, derived directly from the 3DGS rendering equation,
that precisely measures each Gaussian’s contribution to the
rendered image. By introducing a highly efficient algorithm,
our framework enables practical error calculation in a sin-
gle forward pass. The framework is both accurate and flexi-
ble, supporting on-training pruning as well as post-training
simplification via iterative error re-quantification for im-
proved stability. Experimental results show that our method
consistently outperforms existing state-of-the-art pruning
methods across both application scenarios, achieving a su-
perior trade-off between model compactness and high ren-
dering quality.
1. Introduction
3D Gaussian Splatting (3DGS) [11] achieves real-time,
photorealistic novel view synthesis using millions of
anisotropic 3D Gaussian primitives. Despite its impressive
results, the resulting dense representation requires substan-
tial memory and storage. Simplifying 3DGS models by re-
ducing the number of Gaussians is a critical research area,
1
arXiv:2602.06830v1  [cs.CV]  6 Feb 2026

<!-- page 2 -->
Methods
Pruning Criteria
3DGS [11]
α (opacity)
Compact3DGS [13]
Lm = P
α,s(m)
MaskGaussian [18]
Lm = (P Mi)2
GaussianSpa [32]
mina,Θ L(a, Θ), s.t. ||a||0 ≤κ
C3DGS [21]
S(p) = | ∂E
∂p |
Mini-Splatting [5]
Ii = P wij, wij = Tijαij
LightGaussian [4]
GSj = P I(Gj, ri) · αj · T · γ(Σj)
MesonGS [28]
Ig = Ii · (Vnorm)β
PUP-3DGS [9]
Ui = log |∇xi,siIG∇xi,siIT
G |
Ours
∆SEk = ∥Tkαk∆ck∥2
Table 1. Comparison of pruning criteria employed by various 3D
Gaussian Splatting simplification methods.
as this count is the primary factor directly influencing model
size, rendering performance, and deployment feasibility.
Current simplification approaches often evaluate a Gaus-
sian for pruning using importance-based scores [5, 28].
These methods typically employ rendering-based scores,
such as accumulated blending or gradient sensitivity [1, 4,
8], to estimate the visual error rather than to quantify the er-
ror itself. Relying on these metrics that lack a direct corre-
lation with visual impact often yields suboptimal trade-offs
between compactness and fidelity. Therefore, a clear need
exists for a simplification technique that is both principled
in evaluating visual error and flexible in its application. We
propose GaussianPOP, a simplification framework centered
on an analytically-derived criterion that directly quantifies
the visual error of removing a Gaussian. Through extensive
experiments, we show that our method consistently outper-
forms existing state-of-the-art simplification techniques fo-
cused on reducing the number of Gaussians, achieving a su-
perior trade-off between model compactness and rendering
quality. Our contributions are as follows:
• We derive a principled error criterion that analytically
quantifies the pixel error from removing a single Gaus-
sian, providing a more efficient and accurate pruning cri-
terion.
• We introduce a highly efficient algorithm to compute this
error for all Gaussians in a scene, integrated seamlessly
into the 3DGS pruning process.
• We demonstrate the flexibility of our framework as shown
in Fig. 1, enabling it to function both as an on-training-
time pruning schedule and as a post-training simplifica-
tion tool.
2. Related Work
Existing approaches to reduce the number of Gaussians in
3DGS models can be grouped into two categories.
The
first category, herein referred to as criterion-based prun-
ing, evaluates individual Gaussians using importance-based
metrics.
Examples include opacity-based [11, 12, 24],
gradient-sensitivity (e.g, C3DGS [21] and PUP [9]), or
based on blending contribution and alpha-transmittance
[5, 8, 19, 22, 28]. The key limitation of the approaches
of this first category is their reliance on surrogate scores
(e.g., Mini-Splatting [5, 6] and LightGS [4]), which are not
perfect estimates of the visual quality impact when remov-
ing the associated Gaussians. Therefore, these criteria often
overlook the actual visual error resulting from the removal
of a Gaussian. For instance, a Gaussian with a large scale
might seem important but could be completely occluded
and thus visually irrelevant. In addition, a Gaussian with
low opacity might seem insignificant, yet be critical for a
necessary semi-transparent effect.
The second category, i.e., sparsity induction, jointly
optimizes for simplification and reconstruction.
This is
achieved either through learnable masks [13, 33] or by
formulating the simplification as constrained optimiza-
tion problems, as can be seen in GaussianSpa [32] and
Maskgaussian [18]. One limitation of this approach is its
lack of generality. By fundamentally changing the training
objective to encourage sparsity, these methods introduce a
compromise: the model is pushed toward a sparse solution,
which can prevent it from converging to the high-fidelity
representation of the original 3DGS model. Furthermore,
these methods are not general-purpose simplification tools,
as they require a specialized training process from scratch
and cannot be applied to pre-trained models.
Compression techniques also contribute to reducing the
size of 3DGS representations. Distinct from methods that
reduce the number of Gaussians, another line of work fo-
cuses on compressing the attributes [2, 10, 23].
These
methods employ techniques such as vector quantization
(VQ) for attributes [14, 15, 27, 29], parameter factorization
[17, 25, 26], or as seen in recent works predicting attributes
from compact latent representations [1, 3, 7, 16, 30, 31]. Al-
though these methods are effective for storage reduction and
can be complementary to pruning, we consider compression
of 3DGS attributes a distinct problem from simplification,
as, for instance, it does not reduce the rendering complexity.
Therefore, this paper focuses on the simplification of 3DGS
specifically based on the reduction of the Gaussian count.
Tab. 1 summarizes the criteria these methods use. As
a criterion-based approach, GaussianPOP benefits from the
high flexibility of this category, allowing it to simplify pre-
trained models. Moreover, our method is analytically de-
rived to directly quantify visual error. As we will demon-
strate, this combination of criterion-based flexibility and
accuracy allows GaussianPOP to consistently outperform
both existing criterion-based methods and the optimized
sparsity-induction models.
2

<!-- page 3 -->
…
…
Foreground Background 
3DGS Point
Forward 
Rendering
Initialization
Quantification
Simplification
Pruned 3DGS
Derivation of the error contribution
Calculations for all views
within a CUDA Kernel
Threshold pruning 
(Error magnitude)
More important
Front-to-back    -blending
Foreground
Gaussian 
Background 
Error Quantification
For Gaussian 
Figure 2. Overview of the GaussianPOP simplification pipeline. We quantify the per-pixel error (∆SEk) for each Gaussian by ana-
lytically deriving its visual contribution from the α-blending equation. This error is calculated efficiently, allowing us to prune low-error
Gaussians to create a compact model.
3. Proposed Method
Our method, illustrated in Fig. 2, introduces a principled
error criterion derived directly from the 3DGS rendering
equation. First, we revisit the 3DGS rendering process (Sec.
3.1), then analytically derive the proposed error quantifica-
tion (Sec. 3.2), detail the efficient single-pass algorithm for
its computation (Sec. 3.3), and finally outline the complete
simplification framework (Sec. 3.4).
3.1. Background: 3DGS Rendering
The 3DGS rendering process relies on projecting 3D Gaus-
sians onto the image plane and performing an order-
dependent alpha-blending. For a single pixel, the rendered
color Crender is the composite of all N contributing Gaus-
sians, sorted front-to-back:
Crender =
N
X
k=1
Tkαkck
(1)
where ck is the base color of the k-th Gaussian, αk is its
pixel-specific opacity, and Tk = Qk−1
j=1(1−αj) is the trans-
mittance. In our framework, Crender denotes the final color
rendered by the current set of Gaussians, serving as the
baseline for error quantification. Therefore, the proposed
method does not specifically require ground truth views in
the quantification process. This, like other criterion-based
methods, provides the flexibility to simplify any pre-trained
3DGS model.
3.2. Principled Error Quantification
As discussed in Sec. 2, existing simplification methods typ-
ically rely on importance-based metrics. These criteria of-
ten include accumulating the blending weight or employing
gradient- and sensitivity-based metrics. Since these scores
provide only an indirect measure of visual impact and may
not perfectly reflect it, we formulate a criterion that serves
as a direct approximation of the visual error itself.
We analytically compute the squared error (SE) per-pixel
that would be induced by removing a single Gaussian k
from the sorted list compared to using all Gaussians. Intu-
itively, this error is the squared difference between the orig-
inal color Crender and the new counterfactual color C′
render
that would appear if k was removed. To formalize this, we
can express the original color Crender as the sum of three
parts: (1) the color accumulated before k (Pk−1), (2) the
contribution of k itself Tkαkck, and (3) the contribution of
the background after k (bk+1), which is attenuated by k’s
presence Tk+1bk+1 = Tk(1 −αk)bk+1,
Crender = Pk−1
| {z }
foreground
+ Tkαkck
| {z }
Gaussian k
+ Tk(1 −αk)bk+1
|
{z
}
background
(2)
If we remove Gaussian k, its contribution vanishes, and the
background bk+1 is no longer attenuated by αk. Thus, the
counterfactual color C′
render is just the foreground blending
directly with the background (see Fig. 2),
C′
render = Pk−1
| {z }
foreground
+ Tkbk+1
| {z }
background
(3)
The per-pixel SE induced by the removal of Gaussian k
is ∆SEk = ||Crender −C′
render||2. Substituting the two
expressions above, the foreground terms Pk−1 cancel out,
and this SE calculation simplifies to the difference between
the Gaussian’s color ck and the background color bk+1,
weighted by the Gaussian’s effective contribution Tkαk,
∥Crender −C′
render∥2 = ∥Tkαk (ck −bk+1)∥2
(4)
3

<!-- page 4 -->
Algorithm 1 ‘Render-once’ Efficient Error Computation
(Per-pixel)
▷— Phase 1: Compute list H with forward render-
ing —
1: Crender ∈R3 (The final color of the rendered pixel)
2: H = ⟨(ID, ck, αk)⟩N
k=1 (An ordered list of N con-
tributing Gaussians, sorted front-to-back, where ID is
a global identifier, ck ∈R3 is the k-th base color, and
αk ∈[0, 1] is the opacity)
3: ∆SE (A global array, indexed by Gaussian ID, storing
cumulative error)
4: P (A local array of size N to store cumulative color
sums)
5: T (A local array of size N to store cumulative transmit-
tances)
▷— Phase 2:
Compute Cumulative Color and
Transmittance —
6: Psum ←0
7: T ←1.0
8: for k ←1 to N do
9:
(ID, ck, αk) ←Hk
10:
Psum ←Psum + Tαkck
11:
T ′ ←T(1 −αk)
12:
Pk ←Psum
▷Store Pk = Pk
j=1(Tjαjcj)
13:
Tk ←T ′
▷Store Tk = Qk−1
j=1(1 −αj)
14: end for
▷— Phase 3: Compute Error Contribution based
on Color Difference —
15: for k ←1 to N do
16:
(ID, ck, αk) ←Hk
17:
Pk ←Pk
▷Cumulative color Pk
18:
Tk+1 ←Tk
▷Transmittance after k (Tk+1)
19:
if k = 1 then
20:
Tk ←1.0
21:
else
22:
Tk ←Tk−1
▷Transmittance before k (Tk)
23:
end if
▷Analytically solve for bk+1 from the equation:
Crender = Pk + (Tk+1bk+1)
24:
bk+1 ←(Crender −Pk)/(Tk+1 + ϵ)
▷Calculate the Squared Error (SE) for k (Eq. 4)
25:
∆ck ←ck −bk+1
26:
∆SEk ←∥(Tkαk)∆ck∥2
▷Accumulate ∆SEk to the global Gaussian ID
27:
∆SE[ID] ←∆SE[ID] + ∆SEk
28: end for
However, quantifying the error contribution of each
0
500
1000
1500
2000
SE
10
0
10
1
10
2
10
3
10
4
10
5
10
6
Number of Gaussians (Log Scale)
Distribution of Per-Gaussian Square Error Sum
Figure 3.
Distribution of the cumulative square error ∆SE
across all training views for pre-trained ‘Bonsai’ scene. The
majority of Gaussians contribute near-zero error, indicating that
most Gaussians are non-essential to the final render.
Gaussian presents a significant computational challenge.
A naive approach would require iteratively removing each
Gaussian, one at a time, and re-rendering all training views
to measure the resulting error. For a scene with M Gaus-
sians and V views, this process necessitates M × V full
rendering passes, which is computationally prohibitive.
This bottleneck arises because every rendering pass ne-
cessitates re-fetching the entire set of M Gaussians from
VRAM, which makes naive quantification computationally
intractable.
3.3. Practical Computation of the Error Criterion
We designed our framework specifically to circumvent this
overhead. In contrast to the naive approach requiring M
expensive re-rendering operations, our approach uses a
‘render-once, compute-locally’ strategy, as detailed in Al-
gorithm 1. This strategy begins with Phase 1, a standard
forward rendering pass to generate the ordered list H of
N contributing Gaussians. The algorithm then proceeds in
two distinct computational stages. In Phase 2, a parallel pre-
fix sum operation is executed over H. This pass efficiently
computes the final rendered pixel color Crender and simul-
taneously caches the cumulative color Pk and cumulative
transmittance Tk values. In Phase 3, the kernel leverages
this local cache. This stage is highly parallelizable, as the
error computation for each Gaussian k is independent. It an-
alytically derives the background color bk+1 using only the
final Crender and the cached values Pk, Tk. Here ϵ = 1e−9
is for numerical stability. This enables the ∆SEk calcu-
lation using only values in local registers, which are then
atomically accumulated. This strategy substitutes the in-
tractable re-renders required by a naive approach with com-
putationally parallel local operations.
Consequently, the
quantification stage is highly efficient, making it practical
for both on training-time and post-training simplification.
4

<!-- page 5 -->
Dataset
Mip-NeRF 360
Tanks&Temples
Deep Blending
Method
PSNR↑SSIM↑LPIPS↓#G/M↓PSNR↑SSIM↑LPIPS↓#G/M↓PSNR↑SSIM↑LPIPS↓#G/M↓
3DGS [11]
27.50 0.813
0.221
3.111
23.63 0.850
0.180
1.830
29.42 0.900
0.250
2.780
LP-3DGS-R [33]
27.47 0.812
0.227
1.959
23.60 0.842
0.188
1.244
–
–
–
–
LP-3DGS-M [33]
27.12 0.805
0.239
1.866
23.41 0.834
0.198
1.116
–
–
–
–
Compact3DGS [13]
27.08 0.798
0.247
1.388
23.32 0.831
0.201
0.836
29.79 0.901
0.258
1.060
EAGLES [7]
27.23 0.809
0.238
1.330
23.37 0.840
0.200
0.650
29.86 0.910
0.250
1.190
CompGS [20]
27.12 0.806
0.240
0.845
23.44 0.838
0.198
0.520
29.90 0.907
0.251
0.550
Mini-Splatting [5]
27.40 0.821
0.219
0.559
23.45 0.841
0.186
0.319
30.05 0.909
0.254
0.397
LightGaussian* [4]
26.81 0.788
0.273
0.451
23.10 0.804
0.228
0.251
27.57 0.824
0.298
0.302
MaskGaussian* [18]
26.95 0.802
0.252
0.430
23.57 0.837
0.200
0.384
29.80 0.905
0.257
0.342
GaussianSpa [32]
27.25 0.812
0.225
0.426
23.40 0.841
0.188
0.295
29.98 0.910
0.252
0.262
Proposed (ours)
27.36 0.816
0.223
0.401
23.64 0.844
0.186
0.252
29.95 0.909
0.252
0.275
GaussianSpa* [32]
27.43 0.821
0.219
0.353
23.61 0.848
0.183
0.185
30.11 0.911
0.250
0.216
Proposed (ours)*
27.51 0.822
0.218
0.330
23.75 0.850
0.181
0.180
30.18 0.912
0.249
0.208
Table 2. Quantitative results on multiple datasets, compared with 3DGS pruning models (On-training time pruning). * denotes
40,000 training iterations, used to enable a direct comparison with the full GaussianSpa configuration. #G/M represents the number of
Gaussians in millions.
Fig. 3 shows the distribution of the cumulative ∆SE val-
ues, computed across all training views, on a log scale. The
histogram clearly illustrates a long-tailed distribution, with
the majority of Gaussians contributing to error values close
to zero when being removed and very few leading to high
error values. This suggests the existence of numerous re-
dundant Gaussians that contribute little (∆SE ≈0) to the
final rendered image.
This distribution is a key motiva-
tion for our proposed pruning strategy, as it implies that we
can identify and remove a large portion of these Gaussians
while minimizing visual quality degradation, i.e., by remov-
ing those that would lead to very low error values.
3.4. Simplification Framework
The GaussianPOP framework leverages its efficient error
metric in a three-stage process: quantification, pruning,
and fine-tuning. First, the quantification stage executes our
quantification Algorithm 1 over the complete set of train-
ing views. This computes the cumulative ∆SEk for ev-
ery Gaussian in the scene, populating a global error buffer,
∆SE. Then, the pruning stage sorts all Gaussians globally
by their accumulated ∆SE score. Then, a target percentage
P% of the Gaussians with the lowest error score is pruned.
This on-the-fly operation does not rely on backward gra-
dients, unlike mask-based or sparsity losses. Finally, the
fine-tuning stage allows the model to compensate for the
removed Gaussians. The quantification strategy adapts to
the specific application scenario. The on-training applica-
tion integrates our metric as a pruning schedule applied at
discrete steps during training. In contrast, post-training ap-
plication applies a more accurate iterative re-quantification,
an approach analyzed in detail in Sec. 4.3.
4. Experiments and Results
We present a comprehensive evaluation to validate Gaus-
sianPOP. First, we detail our experimental setup (Sec. 4.1).
Second, we benchmark our method against state-of-the-
art simplification techniques in both on-training and post-
training pruning scenarios (Sec. 4.2). Then, we provide ab-
lation studies on our iterative quantification strategy (Sec.
4.3), and analyze the computational complexity (Sec. 4.4).
We evaluate our method against two distinct categories of
3DGS simplification methods: on-training and post-training
pruning methods. Despite being a criterion-based pruning
method, our framework offers high flexibility while outper-
forming existing state-of-the-art methods in both scenarios.
4.1. Experimental Conditions
On-training Pruning Conditions. We integrate our frame-
work directly into the training process from scratch. To en-
sure a fair comparison, we followed the configuration from
GaussianSpa, including its mini-splatting densification [5].
Our pruning mechanism is applied twice during training: at
15k and 20k iterations for a 30k total training iteration, and
at 20k and 25k iterations for a 40k total training iteration.
At these points, our error criterion ∆SE is computed, and
a percentage of the Gaussians with the lowest error score is
permanently removed. The training then continues, allow-
ing the remaining Gaussians to optimize their parameters
and compensate for the removal.
5

<!-- page 6 -->
GaussianSpa
Ours
3DGS
25.95dB/0.25M
27.49dB/4.14M
26.22dB/0.25M
Figure 4. Qualitative comparison for ‘Garden’ scene at 0.25M Gaussians. The density visualization (top row) illustrates our method
achieves a more effective distribution for capturing detailed regions compared to GaussianSpa. This superior distribution results in a final
render (bottom row) that preserves finer detail and texture, achieving a higher PSNR.
0.10M
0.20M
0.30M
0.40M
0.50M
Number of Gaussians
27
28
29
30
31
32
PSNR (dB)
Bonsai
Method
Proposed
GaussianSpa
LightGaussian
MaskGaussian
0.20M
0.30M
0.40M
0.50M
0.60M
Number of Gaussians
27.5
28.0
28.5
29.0
PSNR (dB)
Counter
Method
Proposed
GaussianSpa
LightGaussian
MaskGaussian
Figure 5. Quality-compactness trade-off for on-training prun-
ing. The plots compare PSNR against the number of Gaussians
for our method and competing approaches on two representative
scenes.
Post-training Pruning Conditions.
We start from a
fully pre-trained 3DGS model with 30k iterations.
We
evaluate the pruned model’s performance immediately af-
ter pruning (post-prune) and after an optional 5k iterations
(fine-tuning).
For these experiments, we apply our cri-
terion using 8 iterative cycles, as described in Sec. 4.3.
One advantage enabling the post-prune scenario is that our
error quantification does not require ground truth views,
like other criterion-based methods. This differs from spar-
sity induction models [18, 32], which is integrated into the
training loop and require continuous access to ground truth
views for their loss calculation and the subsequent, manda-
tory fine-tuning process.
4.2. Experimental Results
We evaluate our method as both an on-training time prun-
ing schedule and a post-training simplification tool. The
proposed method consistently outperforms leading meth-
ods from both approaches, achieving a superior trade-off
between model compactness and rendering quality.
On-training Time Pruning Evaluations. Despite be-
ing a criterion-based pruning method, our method consis-
tently outperforms sparsity induction approaches, as shown
in Tab. 2. In the 30k iteration configuration, our method al-
ready demonstrates superior efficiency by achieving better
rendering quality with a more compact model. This perfor-
mance advantage is still clear in the 40k iteration configura-
tion for full GaussianSpa model, where our model achieves
a higher PSNR with a substantially smaller Gaussian count
compared to the equivalent 40k GaussianSpa model. This
consistent performance across all benchmarks validates our
principled error quantification as a more effective simplifi-
cation strategy, one that more accurately identifies and re-
moves visually redundant information.
Fig. 4 provides a visual confirmation of this quantitative
advantage. When comparing the models pruned to an iden-
tical 0.25M Gaussian count, the render from GaussianSpa
exhibits a noticeable loss of texture and detail. Our method,
at the same Gaussian count, produces a visibly sharper and
more stable visual result. This provides direct visual evi-
dence that our error metric is more effective at distinguish-
ing visually critical details from redundant information than
the heuristics used by competing methods. Fig. 5 shows
the quality–compactness trade-off, where our method con-
sistently defines a superior convex hull, achieving a higher
PSNR for any given Gaussian budget. This result validates
the quantitative findings, visually demonstrating the effi-
ciency of our pruning criterion over competing approaches.
6

<!-- page 7 -->
40%
50%
60%
70%
80%
90%
Pruning ratio (%)
15
20
25
30
PSNR (dB)
Bonsai
Method & Fine tuning iteration (T)
Proposed (POP)
PUP3DGS
C3DGS
MesonGS
LightGS
MaskGaussian*
 Post-prune (T=0)
 Finetuning (T=5000)
40%
50%
60%
70%
80%
90%
Pruning ratio (%)
17.5
20.0
22.5
25.0
27.5
PSNR (dB)
Counter
Method & Fine tuning iteration (T)
Proposed (POP)
PUP3DGS
C3DGS
MesonGS
LightGS
MaskGaussian*
 Post-prune (T=0)
 Finetuning (T=5000)
Figure 6. Post-training pruning performance on the ‘Bonsai’
and ‘Counter’ scenes. We compare our method against various
heuristic-based criteria across a range of pruning ratios.
Post-training Time Pruning Evaluations. To specifi-
cally analyze the effectiveness of our ∆SEk error criterion,
we evaluate our method’s performance as a post-training
tool. First, we compare our method against other pruning
criteria [4, 9, 21] across a range of pruning ratios on the
‘Bonsai’ and ‘Counter’ scenes. The results, in Fig. 6, show
that even without fine-tuning (dashed lines), our method
maintains a significantly higher PSNR than all competi-
tors, especially at high pruning ratios. This validates that
our metric is inherently more accurate at identifying non-
essential Gaussians.
When fine-tuning (solid lines, 5k iterations) is applied,
our method’s performance shows remarkable stability and
recovery, demonstrating that our criterion preserves a ro-
bust subset of Gaussians that adapt effectively.
We also
conducted a direct qualitative and quantitative comparison
on the ‘Bicycle’ scene, pruning 80% of Gaussians from a
pre-trained 3DGS model. As seen in Fig. 7, our method
preserves fine details more effectively than the pruning cri-
terion of LightGaussian [4], demonstrating a distinct advan-
(a) LightGaussian (80% Pruned)
(b) Ours (80% Pruned)
Figure 7. Qualitative comparison of post-prune results on the
‘Bicycle’ scene. Both models are pruned by 80% from a pre-
trained 3DGS model. Our method preserves fine details (see bi-
cycle spokes) more effectively than LightGaussian [4].
Method
PSNR→(+FT)
SSIM→(+FT)
LPIPS→(+FT)
#G/M
LightGS [4]
18.46 →24.35
0.537 →0.713
0.364 →0.272
0.97
C3DGS [21]
19.15 →24.39
0.565 →0.719
0.352 →0.267
0.97
PUP3DGS [9]
18.01 →24.37
0.538 →0.714
0.366 →0.264
0.97
MaskGaussian* [18]
N/A
24.88
N/A
0.739
N/A
0.255
1.02
GaussianSpa* [32]
N/A
24.79
N/A
0.740
N/A
0.257
0.97
Ours (POP)
20.71 →25.11
0.616 →0.747
0.332 →0.251
0.97
Table 3.
Quantitative comparison of post-training pruning
methods on the ‘Bicycle’ scene. We report performance with-
out fine-tuning (Post-prune) and with a 5,000 iteration fine-tuning
(+FT) from the pruned model.
tage in reconstructing thin, high-frequency structures. As
shown in Tab. 3, with 80% pruning and post-prune (T=0),
our method achieves a PSNR of 20.71dB, outperforming
LightGaussian’s 18.46dB. After a 5k fine-tuning (+FT), our
method maintains its lead, demonstrating the robustness of
our ∆SEk criterion.
It is important to note that sparsity induction methods
(denoted *), such as GaussianSpa [32] and MaskGaussian
[18], are excluded from the evaluations in the post-prune
scenario. Because their simplification process is inherently
tied to joint training, they do not provide a standalone prun-
ing metric that can be applied to a pre-trained model. While
those methods rely on joint training with sparsity loss to
perform pruning every 1000 iterations, our method achieves
superior performance even without modifying the 3DGS
training process.
4.3. Iterative Error Quantification
Post-training simplification has practical value for real-
world deployment as it enables the immediate compaction
of pre-trained models without the prohibitive computational
overhead of re-training from scratch.
This allows users
to efficiently optimize existing legacy 3DGS models for
resource-constrained devices. However, post-training sim-
plification applies pruning to a static, fully-trained model,
typically involving the removal of a large percentage of
Gaussians all at once.
This difference necessitates a
more precise error quantification to avoid significant quality
degradation.
7

<!-- page 8 -->
Scene
Re-quantifying
Cycle
Total ∆SE
Threshold
Actual PSNR Error (dB)
Training views (Test views)
#G/M
Bonsai
C = 1
1500
-2.98 (-1.69)
0.46
C = 2
-2.13 (-1.13)
0.47
C = 4
-1.56 (-0.83)
0.48
C = 8
-1.14 (-0.66)
0.48
Playroom
C = 1
2500
-2.53 (-0.78)
0.79
C = 2
-1.94 (-0.60)
0.77
C = 4
-1.57 (-0.48)
0.77
C = 8
-1.39 (-0.43)
0.77
Kitchen
C = 1
2500
-2.30 (-1.35)
0.75
C = 2
-1.84 (-1.01)
0.76
C = 4
-1.53 (-0.86)
0.76
C = 8
-1.34 (-0.78)
0.77
Stump
C = 1
5000
-3.06 (-0.53)
1.61
C = 2
-2.26 (-0.45)
1.67
C = 4
-1.90 (-0.39)
1.74
C = 8
-1.75 (-0.36)
1.78
Table 4. Analysis of the effects of iterative re-quantification.
We compare the impact of increasing the re-evaluation cycle C,
which significantly reduces the actual PSNR error, validating the
robustness of the iterative approach.
Our ∆SEk criterion, being a direct measure of visual er-
ror, enables budget-based pruning. We set an absolute total
error threshold B and remove all Gaussians whose ∆SEk
falls below it. However, a naive one-shot calculation of this
budget is a greedy approximation. The challenge lies in
the α-blending process, where removing a large batch of
Gaussians simultaneously affects the transmittance Tj for
all subsequent Gaussians j > k. This cascading effect is
not captured by the single-pass ∆SEk calculation. Con-
sequently, this one-shot quantification tends to underesti-
mate the errors, leading to a cumulative visual impact that
is larger than the expected error budget. This highlights the
necessity of our iterative re-quantification approach.
We analyze an iterative re-quantification approach, as
shown in Tab. 4. Negative PSNR error denotes PSNR de-
crease relative to the unpruned baseline. Instead of apply-
ing the total budget B at once, we divide it across C itera-
tive cycles, applying only a partial budget B/C per cycle.
Crucially, we re-calculate the error ∆SEk for all remaining
Gaussians at the start of each new cycle. This iterative cycle
ensures error metrics are updated to reflect visual changes
from previous removals, leading to a much more accurate
and stable simplification. The results demonstrate the clear
superiority of this iterative approach.
A single-pass re-
moval results in the largest actual PSNR error, highlight-
ing the one-shot limitation. In contrast, by re-evaluating the
∆SEk values after redundant low ∆SEk Gaussians are re-
moved, the error calculation is more accurate than a single-
pass quantification. Consequently, this approach achieves
a more accurate pruning, significantly reducing the actual
PSNR estimation error and better aligning the actual train-
ing view error with the intended error budget.
Method
Pruning Iteration
Total Training Time
#G/M
3DGS (Bicycle)
N/A
27m11s
4.87
Ours (30k), P = [0.5, 0.5]
[15000, 20000]
24m49s
1.23
GaussianSpa (30k), κ = [0.5, 0.5]
[15000, 20000]
25m33s
1.23
MaskGaussian (30k), λm = 0.1
Every 1000
26m21s
1.45
(a) On-training time complexity for ‘Bicycle’ scene. A comparison of the
total training time of our method with two pruning steps against the 3DGS
baseline.
Method
Quantification Time Cost
PSNR→(+FT)
#G/M
C = 1, P = 0.9
4.76s/iter (Total 4.76s)
17.50dB→23.12dB
0.18
C = 2, P = 0.9
4.45s/iter (Total 8.89s)
18.92dB→23.35dB
0.18
C = 4, P = 0.9
4.01s/iter (Total 16.0s)
19.53dB→23.43dB
0.18
C = 8, P = 0.9
3.83s/iter (Total 30.6s)
19.62dB→23.54dB
0.18
(b) Post-training iterative re-evaluation for ‘Tanks&Temples’ dataset.
Analysis of computational cost and visual quality for different re-evaluation
cycles.
Table 5. Analysis of computational complexity for each sce-
nario. (a) On-training time scenario and (b) Post-training iterative
re-evaluation scenario.
4.4. Complexity Analysis
We analyze the computational cost of our quantification
stage, benchmarked on an NVIDIA A100 GPU, with re-
sults detailed in Tab. 5. In the on-training scenario, our
method can reduce the total training time compared to the
original 3DGS model for 30k training. This acceleration
occurs because pruning significantly reduces the Gaussian
count, which accelerates all subsequent training iterations.
The overhead of the quantification itself is negligible, re-
quiring only 10.52s for two quantification passes.
In the post-training scenario, we applied different iter-
ative cycles, and the total cost increases with the number
of cycles. However, the cost per iteration progressively de-
creases, from 4.76s on the first pass to 3.83s. This demon-
strates the substantial quality gains clearly justify the min-
imal overhead, as the model achieves 90% compactness
from the original pre-trained model (23.62dB/1.812M) with
only a 0.1dB PSNR drop after 5k fine-tuning.
5. Conclusion
We presented GaussianPOP, a principled simplification
framework that analytically quantifies the visual error of
each Gaussian directly from the rendering equation. Our
method introduces an efficient single-pass algorithm that
makes this quantification practical with minimal over-
head. The error-based criterion supports both on-training
pruning and post-training simplification via iterative re-
quantification, which further improves stability and reduces
cumulative visual degradation.
Consequently, Gaussian-
POP provides a robust and general method for compact
representation, showing a superior trade-off between model
compactness and rendering fidelity. These results under-
score the advantages of direct error quantification over con-
ventional importance scores for 3DGS simplification.
8

<!-- page 9 -->
References
[1] Muhammad
Salman
Ali,
Sung-Ho
Bae,
and
Enzo
Tartaglione.
ELMGS: enhancing memory and compu-
tation scalability through compression for 3d gaussian
splatting. In IEEE/CVF Winter Conference on Applications
of Computer Vision, WACV 2025, Tucson, AZ, USA, Febru-
ary 26 - March 6, 2025, pages 2591–2600. IEEE, 2025.
2
[2] M. T. Bagdasarian, P. Knoll, Y. Li, F. Barthel, A. Hilsmann,
P. Eisert, and W. Morgenstern. 3dgs.zip: A survey on 3d
gaussian splatting compression methods. Computer Graph-
ics Forum, 44(2):e70078, 2025. 2
[3] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai.
Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision, 2024. 2
[4] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, and Zhangyang Wang. Lightgaussian: unbounded 3d
gaussian compression with 15x reduction and 200+ fps. In
Proceedings of the 38th International Conference on Neural
Information Processing Systems, Red Hook, NY, USA, 2024.
Curran Associates Inc. 2, 5, 7
[5] Guangchi Fang and Bing Wang. Mini-splatting: Represent-
ing scenes with a constrained number of gaussians. In Com-
puter Vision – ECCV 2024: 18th European Conference, Mi-
lan, Italy, September 29–October 4, 2024, Proceedings, Part
LXXVII, page 165–181, Berlin, Heidelberg, 2024. Springer-
Verlag. 2, 5
[6] Guangchi Fang and Bing Wang. Mini-splatting2: Building
360 scenes within minutes via aggressive gaussian densifica-
tion, 2024. 2
[7] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. EA-
GLES: efficient accelerated 3d gaussians with lightweight
encodings.
In Computer Vision - ECCV 2024 - 18th Eu-
ropean Conference, Milan, Italy, September 29-October 4,
2024, Proceedings, Part LXIII, pages 54–71. Springer, 2024.
2, 5
[8] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias
Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaus-
sian splatting with sparse pixels and sparse primitives. In
2025 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 21537–21546, 2025. 2
[9] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayaward-
hana, Matthias Zwicker, and Tom Goldstein.
Pup 3d-gs:
Principled uncertainty pruning for 3d gaussian splatting. In
2025 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 5949–5958, 2025. 2, 7
[10] Yuning Huang, Jiahao Pang, Fengqing Zhu, and Dong Tian.
Entropygs: An efficient entropy coding on 3d gaussian splat-
ting, 2025. 2
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(ToG), 42(4):1–14, 2023. 1, 2, 5
[12] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. In Proceedings of the 38th
International Conference on Neural Information Processing
Systems, Red Hook, NY, USA, 2024. Curran Associates Inc.
2
[13] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park.
Compact 3d gaussian representation
for radiance field. In 2024 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 21719–
21728, 2024. 2, 5
[14] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian splatting for static
and dynamic radiance fields, 2024. 2
[15] Joo Chan Lee, Jong Hwan Ko, and Eunbyung Park.
Op-
timized minimal 3d gaussian splatting.
arXiv preprint
arXiv:2503.16924, 2025. 2
[16] Soonbin Lee, Fangwen Shu, Yago Sanchez, Thomas Schierl,
and Cornelius Hellge. Compression of 3d gaussian splatting
with optimized feature planes and standard video codecs,
2025. 2
[17] Xiangrui Liu, Xinju Wu, Pingping Zhang, Shiqi Wang, Zhu
Li, and Sam Kwong. Compgs: Efficient 3d scene represen-
tation via compressed gaussian splatting.
In Proceedings
of the 32nd ACM International Conference on Multimedia,
page 2936–2944, New York, NY, USA, 2024. Association
for Computing Machinery. 2
[18] Yifei Liu, Zhihang Zhong, Yifan Zhan, Sheng Xu, and Xiao
Sun.
Maskgaussian: Adaptive 3d gaussian representation
from probabilistic masks. In 2025 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
681–690, 2025. 2, 5, 6, 7
[19] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre.
Taming 3dgs: High-quality radiance
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers, New York, NY, USA, 2024. Association for
Computing Machinery. 2
[20] K. L. Navaneet, Kossar Pourahmadi Meibodi, Soroush Ab-
basi Koohpayegani, and Hamed Pirsiavash.
Compgs:
Smaller and faster gaussian splatting with vector quantiza-
tion. In Computer Vision - ECCV 2024 - 18th European Con-
ference, Milan, Italy, September 29-October 4, 2024, Pro-
ceedings, Part XXXII, pages 330–349. Springer, 2024. 5
[21] Simon Niedermayr, Josef Stumpfegger, and R¨udiger West-
ermann. Compressed 3d gaussian splatting for accelerated
novel view synthesis.
In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, CVPR 2024, Seattle,
WA, USA, June 16-22, 2024, pages 10349–10358. IEEE,
2024. 2, 7
[22] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakoto-
saona, Michael Oechsle, Daniel Duckworth, Rama Gosula,
Keisuke Tateno, John Bates, Dominik Kaeser, and Federico
Tombari. Radsplat: Radiance field-informed gaussian splat-
ting for robust real- time rendering with 900+ fps. In 2025
International Conference on 3D Vision (3DV), pages 134–
144, 2025. 2
[23] Panagiotis Papantonakis,
Georgios Kopanas,
Bernhard
Kerbl, Alexandre Lanvin, and George Drettakis. Reducing
9

<!-- page 10 -->
the memory footprint of 3d gaussian splatting. Proc. ACM
Comput. Graph. Interact. Tech., 7(1), 2024. 2
[24] St´ephane Pateux, Matthieu Gendrin, Luce Morin, Th´eo
Ladune, and Xiaoran Jiang. Bogauss: Better optimized gaus-
sian splatting, 2025. 2
[25] Seungjoo Shin, Jaesik Park, and Sunghyun Cho. Locality-
aware gaussian compression for fast and high-quality ren-
dering, 2025. 2
[26] Xiangyu Sun, Joo Chan Lee, Daniel Rho, Jong Hwan Ko,
Usman Ali, and Eunbyung Park.
F-3dgs: Factorized co-
ordinates and representations for 3d gaussian splatting. In
Proceedings of the 32nd ACM International Conference on
Multimedia, page 7957–7965, New York, NY, USA, 2024.
Association for Computing Machinery. 2
[27] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jia-
jun Deng, Jiang Bian, and Zhibo Chen.
End-to-end rate-
distortion optimized 3d gaussian representation. In European
Conference on Computer Vision, 2024. 2
[28] Shuzhao Xie, Weixiang Zhang, Chen Tang, Yunpeng Bai,
Rongwei Lu, Shijia Ge, and Zhi Wang.
Mesongs: Post-
training compression of 3d gaussians via efficient attribute
transformation. In Computer Vision – ECCV 2024: 18th Eu-
ropean Conference, Milan, Italy, September 29–October 4,
2024, Proceedings, Part XXXIII, page 434–452, Berlin, Hei-
delberg, 2024. Springer-Verlag. 2
[29] Hao Xu, Xiaolin Wu, and Xi Zhang. Improving 3d gaussian
splatting compression by scene-adaptive lattice vector quan-
tization, 2025. 2
[30] Hao Xu, Xiaolin Wu, and Xi Zhang. 3dgs compression with
sparsity-guided hierarchical transform coding, 2025. 2
[31] Yu-Ting Zhan,
Cheng-Yuan Ho,
Hebi Yang,
Yi-Hsin
Chen, Jui-Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng.
CAT-3DGS: A context-adaptive triplane approach to rate-
distortion-optimized 3dgs compression.
In The Thir-
teenth International Conference on Learning Representa-
tions, ICLR 2025, Singapore, April 24-28, 2025. OpenRe-
view.net, 2025. 2
[32] Yangming Zhang, Wenqi Jia, Wei Niu, and Miao Yin. Gaus-
sianspa: An ”optimizing-sparsifying” simplification frame-
work for compact and high-quality 3d gaussian splatting. In
2025 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 26673–26682, 2025. 2, 5,
6, 7
[33] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang,
Cheng Peng, Rama Chellappa, and Deliang Fan. Lp-3dgs:
learning to prune 3d gaussian splatting. In Proceedings of the
38th International Conference on Neural Information Pro-
cessing Systems, Red Hook, NY, USA, 2025. Curran Asso-
ciates Inc. 2, 5
10

<!-- page 11 -->
GaussianPOP: Principled Simplification Framework for
Compact 3D Gaussian Splatting via Error Quantification
Supplementary Material
6. Derivation of the Background Term
To formally justify our error criterion, we define the back-
ground term bk+1 as the color of the scene occluded by the
first k Gaussians. First, we split the rendering equation into
the accumulated color up to the k-th Gaussian Pk and the
contribution of the remaining Gaussians j > k,
Crender =
N
X
i=1
Tiαici =
k
X
i=1
Tiαici
|
{z
}
Pk
+
N
X
j=k+1
Tjαjcj
The second term represents the attenuated background con-
tribution. Since the transmittance Tj for any j > k inher-
ently includes the accumulated transmittance Tk+1, we fac-
tor Tk+1 out to define the background term,
N
X
j=k+1
Tjαjcj = Tk+1
N
X
j=k+1
Tj
Tk+1
αjcj
|
{z
}
bk+1
Physically, bk+1 represents the unattenuated background
color directly behind the k-th Gaussian. Normalizing by
Tk+1 removes the foreground occlusion, enabling a mean-
ingful comparison between the Gaussian’s color ck and the
background bk+1. Thus, ∆SEk serves as a measure of vi-
sual redundancy, identifying primitives that can be pruned
with minimal color loss due to their high similarity to the
underlying background. Here, the term inside the summa-
tion defines bk+1. Substituting this back into the decompo-
sition yields,
Crender = Pk + Tk+1bk+1
We rearrange this equation to calculate bk+1 using the final
rendered color and accumulated values, avoiding the need
for re-computation in our algorithm,
bk+1 = Crender −Pk
Tk+1
7. Implementation Details
In our implementation, we fixed the maximum number of
contributing Gaussians per pixel (Nmax) to 64. We con-
ducted experiments increasing Nmax to 128 and observed
no meaningful difference in the final pruning results or ren-
dering quality.
For comparison, we employ a two-stage
pruning with P = 0.75 for Mip-NeRF 360 outdoor scenes
and P = {0.65, 0.7} for other datasets depending on num-
ber of points. To benchmark GaussianSpa at a comparable
scale, we modified the ‘pruning ratio2’ from the reference
per-scene configuration.
We evaluated the Mip-NeRF 360 dataset using the pre-
downsampled images (images_2 and images_4) instead of
the automatic downsampling. Due to different downsam-
pling algorithms, these pre-downsampled images preserve
fine details, making the reconstruction task considerably
more difficult. This protocol yields lower PSNR (e.g., 25.6
to 25.2dB for bicycle) for all comparisons but provides a
more accurate evaluation of detail preservation.
8. Per-scene Quantitative Results
Tanks & Temples
Scene
Method
PSNR↑
SSIM↑
LPIPS↓
#G/M↓
Train
3DGS
21.94
0.815
0.210
1.110
Mini-Splatting
21.78
0.805
0.231
0.287
GaussianSpa
21.97
0.815
0.228
0.200
Ours
22.21
0.815
0.224
0.177
Truck
3DGS
25.31
0.885
0.150
2.540
Mini-Splatting
25.13
0.878
0.141
0.352
GaussianSpa
25.25
0.881
0.138
0.170
Ours
25.28
0.885
0.138
0.183
Average
3DGS
23.63
0.850
0.180
1.825
Mini-Splatting
23.46
0.841
0.186
0.320
GaussianSpa
23.61
0.848
0.183
0.185
Ours
23.75
0.850
0.181
0.180
Deep Blending
Scene
Method
PSNR↑
SSIM↑
LPIPS↓
#G/M↓
DrJohnson
3DGS
28.77
0.900
0.250
3.260
Mini-Splatting
29.37
0.904
0.261
0.377
GaussianSpa
29.51
0.909
0.247
0.214
Ours
29.62
0.911
0.246
0.195
Playroom
3DGS
30.07
0.900
0.250
2.290
Mini-Splatting
30.72
0.914
0.248
0.417
GaussianSpa
30.71
0.913
0.253
0.218
Ours
30.74
0.913
0.252
0.221
Average
3DGS
29.42
0.900
0.250
2.775
Mini-Splatting
30.05
0.909
0.254
0.397
GaussianSpa
30.11
0.911
0.250
0.216
Ours
30.18
0.912
0.249
0.208
Table 6. ‘On-training pruning’, Quantitative comparison on
Tanks & Temples and Deep Blending datasets. Note that both
GaussianSpa and our method are trained for 40k iterations.
11

<!-- page 12 -->
Table 7. ‘On-training pruning’, Quantitative comparison on
Mip-NeRF 360 dataset. We compare our method with 3DGS,
Mini-Splatting, and GaussianSpa. Note that both GaussianSpa and
our method are trained for 40k iterations.
Scene
Method
PSNR↑
SSIM↑
LPIPS↓
#G/M↓
Bicycle
3DGS
25.13
0.750
0.240
5.310
Mini-Splatting
25.21
0.760
0.247
0.696
GaussianSpa
25.26
0.758
0.257
0.465
Ours
25.08
0.755
0.260
0.395
Bonsai
3DGS
32.19
0.950
0.180
1.250
Mini-Splatting
31.73
0.945
0.180
0.360
GaussianSpa
31.92
0.945
0.180
0.287
Ours
32.05
0.947
0.178
0.265
Counter
3DGS
29.11
0.910
0.180
1.170
Mini-Splatting
28.53
0.911
0.184
0.308
GaussianSpa
28.85
0.917
0.178
0.314
Ours
29.01
0.919
0.176
0.325
Flowers
3DGS
21.37
0.590
0.360
3.470
Mini-Splatting
21.42
0.616
0.336
0.670
GaussianSpa
21.57
0.609
0.335
0.356
Ours
21.55
0.605
0.339
0.330
Garden
3DGS
27.32
0.860
0.120
5.690
Mini-Splatting
26.99
0.842
0.156
0.788
GaussianSpa
26.72
0.838
0.159
0.377
Ours
26.81
0.840
0.158
0.345
Kitchen
3DGS
31.53
0.930
0.120
1.770
Mini-Splatting
31.24
0.929
0.122
0.438
GaussianSpa
31.53
0.934
0.117
0.316
Ours
31.75
0.936
0.115
0.325
Room
3DGS
31.59
0.920
0.200
1.500
Mini-Splatting
31.44
0.929
0.193
0.394
GaussianSpa
31.46
0.928
0.190
0.302
Ours
31.50
0.932
0.186
0.288
Stump
3DGS
26.73
0.770
0.240
4.420
Mini-Splatting
27.35
0.803
0.219
0.717
GaussianSpa
26.88
0.806
0.222
0.377
Ours
27.04
0.808
0.218
0.365
Treehill
3DGS
22.61
0.640
0.350
3.420
Mini-Splatting
22.69
0.652
0.332
0.663
GaussianSpa
22.71
0.655
0.335
0.387
Ours
22.79
0.655
0.335
0.335
Average
3DGS
27.50
0.813
0.221
3.111
Mini-Splatting
27.40
0.821
0.219
0.559
GaussianSpa
27.43
0.821
0.219
0.353
Ours
27.51
0.822
0.218
0.330
Table 8. ‘Post-training pruning’, Quantitative comparison on
all datasets. Fine-tuning is performed for 5,000 iterations with
C = 8.
Scene
Threshold
PSNR↑
SSIM↑
LPIPS↓
#G/M↓
Bicycle
P = 0.7
25.17
0.751
0.249
1.460
P = 0.8
25.12
0.747
0.252
0.973
P = 0.9
24.32
0.672
0.350
0.487
Bonsai
P = 0.7
32.18
0.941
0.213
0.321
P = 0.8
32.09
0.936
0.223
0.214
P = 0.9
30.98
0.914
0.262
0.107
Counter
P = 0.7
29.14
0.908
0.207
0.324
P = 0.8
29.00
0.902
0.219
0.216
P = 0.9
28.40
0.875
0.263
0.108
Flowers
P = 0.7
21.58
0.602
0.354
0.876
P = 0.8
21.51
0.589
0.375
0.584
P = 0.9
21.08
0.542
0.428
0.292
Garden
P = 0.7
27.39
0.863
0.120
1.243
P = 0.8
27.15
0.849
0.144
0.828
P = 0.9
26.28
0.799
0.221
0.414
Kitchen
P = 0.7
31.57
0.927
0.131
0.480
P = 0.8
31.38
0.922
0.141
0.320
P = 0.9
30.56
0.902
0.179
0.160
Room
P = 0.7
31.72
0.921
0.224
0.392
P = 0.8
31.62
0.918
0.230
0.261
P = 0.9
31.26
0.907
0.259
0.131
Stump
P = 0.7
26.70
0.776
0.227
1.285
P = 0.8
26.73
0.776
0.233
0.857
P = 0.9
26.36
0.747
0.287
0.428
Treehill
P = 0.7
22.60
0.630
0.363
0.975
P = 0.8
22.53
0.604
0.413
0.650
P = 0.9
22.13
0.526
0.503
0.325
Train
P = 0.7
22.13
0.813
0.214
0.326
P = 0.8
22.06
0.802
0.237
0.218
P = 0.9
21.80
0.768
0.292
0.109
Truck
P = 0.7
25.39
0.884
0.152
0.616
P = 0.8
25.34
0.879
0.161
0.411
P = 0.9
25.16
0.859
0.205
0.205
Playroom
P = 0.7
30.18
0.909
0.251
0.554
P = 0.8
30.19
0.909
0.255
0.369
P = 0.9
30.11
0.906
0.270
0.185
DrJohnson
P = 0.7
29.46
0.906
0.243
0.938
P = 0.8
29.47
0.906
0.245
0.625
P = 0.9
29.38
0.901
0.261
0.313
12

<!-- page 13 -->
3DGS (25.21dB)
4.79M Gaussians
Ours (25.08dB)
0.39M Gaussians
3.26M Gaussians
Ours (29.42dB)
0.19M Gaussians
4.42M Gaussians
Ours (26.81dB)
0.39M Gaussians
1.12M Gaussians
Ours (22.05dB)
0.17M Gaussians
3DGS (29.47dB)
3DGS (26.75dB)
3DGS (22.08dB)
Figure 8. Qualitative comparison of ‘on-training pruning’ on representative scenes. We compare our method with the 3DGS baseline
across different datasets. Each subfigure reports the rendering quality (PSNR) and the number of Gaussians. Please zoom in for a detailed
view.
13

<!-- page 14 -->
LightGS (80%, Post-prune)
Ours (80%, Post-prune)
LightGS (80%, Fine-tuning)
Ours (80%, Fine-tuning)
LightGS (80%, Post-prune)
Ours (80%, Post-prune)
LightGS (80%, Fine-tuning)
Ours (80%, Fine-tuning)
LightGS (80%, Post-prune)
Ours (80%, Post-prune)
LightGS (80%, Fine-tuning)
Ours (80%, Fine-tuning)
LightGS (80%, Post-prune)
Ours (80%, Post-prune)
LightGS (80%, Fine-tuning)
Ours (80%, Fine-tuning)
Figure 9. Qualitative comparison of ‘post-training pruning’ on representative scenes. We compare our method against LightGS with
an 80% pruning ratio on selected benchmark scenes. ‘Post-prune’ denotes results immediately after pruning, while ‘Fine-tuning’ shows
results after 5,000 refinement iterations. Please zoom in for a detailed view.
14
