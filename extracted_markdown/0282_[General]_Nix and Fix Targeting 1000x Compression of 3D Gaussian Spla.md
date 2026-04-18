# NIX AND FIX: TARGETING 1000Ã COMPRESSION OF 3D GAUSSIAN SPLATTING WITH DIFFUSION MODELS

Cem Eteke 1,2 and Enzo Tartaglione 2

1 Chair of Media Technology, Munich Institute of Robotics and Machine Intelligence School of Computation, Information, and Technology Technical University of Munich, 80333 Munich, Germany 2LTCI, TelÂ´ ecom Paris, Institut Polytechnique de Paris, France Â´

## ABSTRACT

3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifactaware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000Ã rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.

Index Termsâ 3DGS compression, image restoration, diffusion models

## 1. INTRODUCTION

Utilizing 3D Gaussian Splatting (3DGS) for novel-view rendering has emerged as an alternative to implicit neural radiance models [1]. Instead of the dense prediction scheme of the latter, i.e., estimating color and opacity from given points in 3D space, 3DGS fits sparse Gaussians with attributes, such as color, scale, and position, in relevant, non-empty regions. This representation, combined with parallelizable rasterization, enabled real-time novel-view rendering.

The real-time nature of 3DGS is highly relevant to applications such as immersive communications. Nevertheless, the introduction of Gaussians with attributes, i.e., primitives, increases the rate significantly compared to a low-parameter implicit neural network. To alleviate this issue, 3DGS compression has garnered significant research interest in recent years. Proposed methodologies range from unstructured approaches, such as pruning and quantization, to structured approaches, including anchor and graph-based methods [2].

<!-- image-->  
Fig. 1: Artifacts resulting from 3DGS compression at different rates. Rate control is achieved through pruning (Ã), quantization, and entropy coding. Notice the loss of geometry, texture, and radiance.

Even though 3DGS compression achieves up to 100Ã rate gain over a baseline 3DGS [3], at extremely low rates, through the degradation of the underlying 3D representation, rendering introduces complex artifacts. This is unlike 2D image degradation, such as downsampling, as it results from aggregating the distorted 3D representations. Resulting artifacts can include, but are not limited to, blur, loss, and degradation of texture, geometry, and radiance. Figure 1 displays examples of such rendering artifacts for a scene at extreme low rates. Restoring these artifacts would significantly improve the rendering performance at such low rates, thus enabling the implementation of 3DGS in data-constrained applications. This perspective of restoring for operation at extremely low rates is underexplored in the literature.

Although deep image restoration methods have been successful in 2D restoration scenarios, as discussed, 3DGS compression artifacts pose a greater challenge. To this end, generative perceptual restoration represents a promising direction, as pretrained Latent Diffusion Models provide strong natural-image priors and achieve high perceptual quality in data-driven restoration even under severe distortion [4].

We introduce NiFi, which enables extreme 3DGS compression through variational diffusion distillation for one-step restoration [5]. We perform data-driven artifact-aware distillation through a synthetic 3DGS compression dataset. Rather than directly restoring from degraded input, we reparameterize the inference by mapping the image onto an immediate diffusion step, allowing the model to exploit stochastic diversity. We compare our approach with classical [6], deep learning-based [7], generative [8], and 3DGS restoration [9] methods. We report state-of-the-art restoration performance and an almost 1000Ã reduction in rate, with perceptual performance comparable to the non-compressed 3DGS. Our contributions therefore are the following:

â¢ We propose a compression framework for 3DGS enabling access to extremely low bitrates, and perceptual quality can be trade-offed with decoding complexity (Sec. 3)

â¢ We conduct extensive comparisons against classical, deep learning-based, and generative restoration methods, demonstrating state-of-the-art performance under extreme compression (Sec. 4.2).

## 2. RELATED WORK

3D Gaussian Splatting. Kerbl et al. used 3DGS to represent a scene with a set of sparse Gaussians ${ \mathcal G } : = \{ G _ { i } ( \bar { x } ) \} _ { i = 1 } ^ { L } ,$ i.e., primitives [1]. In this formulation, each primitiveâs mean and covariance represent its three-dimensional geometry: the mean encodes position, while the covariance captures rotation and scale. Novel view rendering is achieved by ordering the primitives with respect to a given camera view and projecting them onto the image plane. The pixel color values are then rendered using Î±-compositing of color and opacity, where the color is represented using view-dependent spherical harmonic coefficients. This extends the primitives to include color and opacity as additional attributes: storing these results in high space demand, hindering data-constrained applications.

Towards More Efficient Representations. One dominant approach to efficient 3DGS representation is to reduce the number of primitives, namely, by pruning. Pruning methods remove primitives or regulate densification, reducing the overall size. The pruning can be explicitly carried out in accordance with various measures. For instance, CompGS utilized size and opacity [10]. In addition to these static criteria, many approaches rely on gradient-based measures to guide pruning masks. EfficientGS evaluated which primitives are close to convergence based on gradients of positions and applied an adaptive densification strategy [11]. To further support variable rates, GoDe used gradient-based pruning at multiple rates and finetuning for adaptation [12].

Attribute-based measures for explicit pruning, whether static (e.g., opacity) or dynamic (e.g., gradient), might not be informative enough about rendering performance. Rendering-based pruning methods address this issue. Light-

Gaussian used a scoring-based approach for pruning primitives that do not contribute to the rendering quality [13]. Furthermore, EAGLES introduced an influence metric to prune the primitives with low influence on the rasterization [14]. However, these approaches still rely on explicitly evaluating and removing individual primitives. Alternatively, structured methods such as Scaffold-GS introduce reference points, i.e., anchors, that define dynamic primitives, thereby reducing the number of primitives to store [15].

While pruning and anchoring approaches reduce the number of primitives, the redundancies can be further exploited. To that extent, scalar [12] and vector quantization [14, 13, 10] are combined with entropy coding. To further enable entropy models for 3DGS, HAC++ utilized mutual information among elements in a hash-grid of anchors to create a context model [3]. Furthermore, HEMGS introduced a learnable entropy model and a hyperprior network [16]. CodecGS utilized 2D feature planes and entropy modelling to compress 3DGS with video codecs [17]. While the discussed approaches improve efficiency, at extremely low rates they degrade the underlying 3D representation and introduce complex artifacts. We examined such a case in Fig. 1.

Image and 3D Gaussian Splatting Restoration. Classical methods such as block matching and 3D filtering (BM3D) rely on known degradation models [6]. Wang et al. demonstrated that data-driven training of convolutional neural networks (CNNs) using simulated degradation removes this requirement [18]. Building on this idea, SwinIR extended the approach to the Swin Transformer architecture [7]. More recently, diffusion models (DMs) have achieved perceptually superior restoration quality, surpassing the limits of regression-based methods [19]. Subsequent works have explored hybrid strategies, with DiffBIR combining nongenerative restoration methods and controllable DMs [8], and TSD-SR introducing a reconstruction term into distributionmatching distillation of DMs to enable superresolution [20]. However effective, these approaches cannot generalize to the artifacts introduced by 3DGS, as the degradation of the underlying 3D representation yields complex artifacts beyond the scope of widely studied image restoration tasks.

DMs have been successfully extended to 3DGS novelview artifact restoration. GSFix3D used the backward diffusion to update 3DGS primitives [21]. Difix3D, on the other hand, trained the diffusion backbone for restoration, keeping the primitives fixed [9]. Though they enable better novel-view synthesis, these methods are sensitive to compression artifacts. To account for the artifacts, Shin et al. used a regression-based restoration combined with residual coding for 3DGS compression [22]. However, residual coding increases the rate, and regression-based approaches do not achieve high perceptual quality. In our work, we introduce an extreme 3DGS compression approach that leverages the restoration capabilities of one-step diffusion distillation [20].

<!-- image-->  
Fig. 2: The overall pipeline of NiFi. We create a 3DGS restoration dataset of degraded $\tilde { I }$ and high-quality I frames via Artifact Synthesis, through pruning, quantization, and entropy coding at three rates. $\bar { \tilde { I } }$ is mapped to the latent space and to an intermediate step $t _ { 0 }$ in the diffusion trajectory. ËI is obtained in one step via adapter $\phi ^ { - }$ extending a frozen diffusion backbone $\epsilon _ { \theta }$ for Artifact Restoration. $\phi ^ { - }$ is trained with Restoring Distribution Matching through the critic adapter $\phi ^ { + }$ and Perceptual Matching between ËI and I. Only Artifact Restoration is performed during inference.

## 3. METHODOLOGY

Information loss at the 3D representation distorts both the geometry and appearance, thus compressing 3DGS, especially at extremely low rates, produces complex artifacts. We aim to restore these artifacts by formulating a blind image restoration problem, enabling 3DGS compression at extreme-low rates.

Artifact Synthesis step shown in Fig. 2 generates distorted and clean image pairs by simulating compression artifacts at distinct rates. This simulation approach lies at the core of learning-based blind image restoration methods [18]. To support the simulation of artifacts at a variable rate, we use GoDe [12]. We prunes pretrained 3DGS model at three levels $\left\{ \mathcal { G } _ { 0 } , \mathcal { G } _ { 1 } , \mathcal { G } _ { 2 } \right\}$ , where $\begin{array} { r } { | \mathcal { G } _ { l } | = c _ { m i n } \mathrm { e x p } \left( \frac { \log \left( | \mathcal { G } _ { L - 1 } | \right) - \log \left( | c _ { m i n } | \right) } { L - 1 } \right) } \end{array}$ . $c _ { m i n }$ is the minimum cardinality and $| \mathcal { G } _ { 0 } | = c _ { m i n }$ . The pruning metric is the gradient of the rendering loss with respect to the attributes lowes $\bar { \boldsymbol { \mathscr { \kappa } } } _ { k _ { l } } \| \frac { \partial \mathcal { L } } { \partial \mathcal { G } _ { l } } \| _ { 2 }$ . The 3DGS model is then fine-tuned with pruned versions at three levels, and the resulting attributes are stored using 8-bit quantization and entropy coding. Given a training view I, we finalize the simulation by rendering degraded ËI from a low-rate $\mathcal { G } _ { l }$

We formulate recovering the original frame I given degraded ËI as a distribution matching problem where we match, for a restored estimate ËI, distributions $p _ { \mathrm { r e s t o r e } }$ and $p _ { \mathrm { r e a l } }$

$$
\mathcal { L } _ { K L } = \underset { \hat { I } \sim p _ { \mathrm { r e s t o r e } } } { \mathbb { E } } \left[ \log \left( \frac { p _ { \mathrm { r e s t o r e } } ( \hat { I } ) } { p _ { \mathrm { r e a l } } ( \hat { I } ) } \right) \right] .\tag{1}
$$

To this end, we leverage latent diffusion models (LDM) as the image prior $p _ { \mathrm { r e a l } }$ and aim to learn $p _ { \mathrm { r e s t o r e } }$ [4]. An LDM consists of an encoder $\mathcal { E } ,$ a decoder $\mathcal { D } _ { \mathrm { { ; } } }$ , and a denoiser $\epsilon _ { \theta }$ The LDM serves as the generative latent image distribution by parameterizing the forward diffusion process

$$
x _ { t } = ( 1 - \sigma _ { t } ) x + \sigma _ { t } \varepsilon\tag{2}
$$

where $t \in [ 1 , T ]$ is the diffusion timestep, $\sigma _ { t }$ is the variance

at $t ,$ and $\varepsilon = x _ { T } \sim \mathcal { N } ( 0 , 1 )$ . The diffusion denoising model $\epsilon _ { \theta }$ is trained to perform the inverse diffusion process

$$
x _ { t - 1 } = x _ { t } - ( \sigma _ { t - 1 } - \sigma _ { t } ) \epsilon _ { \theta } ( x _ { t } , t ) .\tag{3}
$$

Artifact Restoration extends $\epsilon _ { \theta }$ with a low-rank adapter $\phi ^ { - }$ to obtain the estimated clean image $\hat { I }$ via the resulting one-step restoration from the model $\epsilon _ { \theta , \phi } .$ â . Furthermore, rather than directly restoring from ${ \tilde { x } } ,$ , we project the latents to an intermediate diffusion state $t _ { 0 }$ using the forward model in Eq. 2, allowing the model to exploit diversity for improved restoration. Hence, we obtain the one-step restored image latents as $\hat { x } = \tilde { x } _ { t _ { 0 } } - \sigma _ { t _ { 0 } } \epsilon _ { \theta , \phi ^ { - } } ( \tilde { x } _ { t _ { 0 } } , t _ { 0 } )$

Restoration Distribution Matching step, illustrated in Fig. 2, to minimizes the KL-Divergence term in Eq. 1 with respect to $\phi ^ { - }$ . To that extend, the variational distillation formulation approximates the intractable, $\nabla _ { \phi ^ { - } } \mathcal { L } _ { K L }$ through

$$
\nabla _ { \phi ^ { - } } \mathcal { L } _ { K L } = \frac { \mathbb { E } } { \hat { x } } \left[ \left( s _ { \mathrm { r e s t o r e } } ( \hat { x } ) - s _ { \mathrm { r e a l } } ( \hat { x } ) \frac { \partial \hat { x } } { \partial \phi ^ { - } } \right) \right]\tag{4}
$$

where $s ( \boldsymbol x ) : = \nabla _ { \boldsymbol x } \log p ( \boldsymbol x )$ is the score, i.e., estimated noisefree image of a diffusion model [5]. For the real image distribution, this is computed as $s _ { \mathrm { r e a l } } ( \hat { x } ) = \hat { x } _ { t } - \sigma _ { t } \epsilon _ { \theta } ( \hat { x } , t )$ . where $t \sim \mathcal { U } [ t _ { m i n } , t _ { m a x } ]$ , sampled during training. The restoration distribution $p _ { \mathrm { r e s t o r e } }$ is modelled with another low-rank adapter that extends frozen $\epsilon _ { \theta }$ with parameters $\phi ^ { + }$ . Hence, $s _ { \mathrm { r e s t o r e } } ( \hat { x } ) = \hat { x } _ { t } - \sigma _ { t } \epsilon _ { \theta , \phi ^ { + } } ( \hat { x } _ { t } , t )$ . The model $\epsilon _ { \theta , \phi ^ { + } }$ enables distribution matching by learning the distribution of the restored frames in addition to the restoration itself.

Though effective for single-step synthesis, this formulation alone does not explicitly guide distillation to restore a high-quality frame while keeping the fidelity. To achieve that, we utilize guidance in the ground-truth direction [20]. Dong et al. extended the distribution matching as

Table 1: Quantitative results at three rates and the single-rate 3DGS baseline. For LPIPS and DISTS, lower is better (â). We highlight best, second-best and third-best performing methods.
<table><tr><td></td><td colspan="3">Mip-NeRF360 [23] (LPIPS â / DISTS â)</td><td colspan="3">Tanks &amp; Temples [24] (LPIPS â / DISTS )</td><td colspan="3">DeepBlending [25] (LPIPS â / DISTS â)</td></tr><tr><td>Size (MB)</td><td colspan="3">576</td><td colspan="3">339</td><td colspan="3">555</td></tr><tr><td>3DGS-30K [1]</td><td colspan="3"></td><td colspan="3">0.125 / 0.067</td><td colspan="3">0.125 / 0.098</td></tr><tr><td>Size (MB)</td><td>1.152</td><td>0.156 / 0.078 0.357</td><td>0.223</td><td>1.312</td><td>0.381</td><td>0.230</td><td>0.599</td><td>0.183</td><td>0.110</td></tr><tr><td>HAC++ [3]</td><td>0.285 / 0.158</td><td>0.409 / 0.201</td><td>0.455 / 0.220</td><td>0.210 / 0.113</td><td>0.317 / 0.169</td><td>0.364 / 0.193</td><td>0.201/ 0.161</td><td>0.282 / 0.215</td><td>0.337 / 0.249</td></tr><tr><td>+ BM3D [6]</td><td>0.308 / 0.164</td><td>0.423 / 0.206</td><td>0.466 / 0.226</td><td>0.225 / 0.124</td><td>0.325 / 0.175</td><td>0.371/ 0.199</td><td>0.209 / 0.172</td><td>0.286 / 0.222</td><td>0.340 / 0.254</td></tr><tr><td>+ SwinIR [7]</td><td>0.346 / 0.218</td><td>0.436 / 0.239</td><td>0.468 / 0.248</td><td>0.281/ 0.159</td><td>0.350 / 0.192</td><td>0.381 /0.209</td><td>0.236 / 0.180</td><td>0.290 / 0.213</td><td>0.333 / 0.237</td></tr><tr><td>+ 2 [26] </td><td>0.341 / 0.156</td><td>0.438 / 0.200</td><td>0.477 / 0.220</td><td>0.299 / 0.128</td><td>0.372 / 0.174</td><td>0.408 / 0.197</td><td>0.246 / 0.169</td><td>0.313 / 0.218</td><td>0.362 / 0.249</td></tr><tr><td>+ DiffBIR [8]</td><td>0.350 / 0.178</td><td>0.417 / 0.203</td><td>0.446 / 0.215</td><td>0.283 / 0.154</td><td>0.338 / 0.179</td><td>0.375 / 0.196</td><td>0.298 / 0.208</td><td>0.368 / 0.228</td><td>0.410 / 0.241</td></tr><tr><td>+ Difix3D [9]</td><td>0.238/ 0.133</td><td>0.300 / 0.145</td><td>0.330 / 0.152</td><td>0.165/ 0.088</td><td>0.235/ 0.116</td><td>0.272/ 0.131</td><td>0.158 / 0.111</td><td>0.216 / 0.143</td><td>0.257 / 0.161</td></tr><tr><td>+ NiFi (Ours)</td><td>0.178/0.109</td><td>0.235/0.133</td><td>0.265/0.153</td><td>0.128/0.076</td><td>0.180/0.095</td><td>0.212/0.109</td><td>0.133/0.101</td><td>0.180/0.131</td><td>0.218/0.156</td></tr><tr><td></td><td>+ NiFi (w/o t0) 0.211 / 0.129</td><td>0.287 / 0.174</td><td>0.324 / 0.197</td><td>0.153 / 0.104</td><td>0.229 / 0.147</td><td>0.269 / 0.173</td><td>0.162/ 0.147</td><td>0.231 / 0.193</td><td>0.282 / 0.223</td></tr></table>

$$
\begin{array} { r l } { \displaystyle \nabla _ { \phi ^ { - } } \mathcal { L } _ { \mathrm { K L } } = \alpha \underset { \hat { x } } { \mathbb { E } } \bigg [ \big ( s _ { \mathrm { r e s t o r e } } ( \hat { x } ) - s _ { \mathrm { r e a l } } ( \hat { x } ) \big ) \frac { \partial \hat { x } } { \partial \phi ^ { - } } \bigg ] + } & { } \\ { \displaystyle + \left( 1 - \alpha \right) \underset { x , \hat { x } } { \mathbb { E } } \bigg [ \left( s _ { \mathrm { r e a l } } ( x ) - s _ { \mathrm { r e a l } } ( \hat { x } ) \right) \frac { \partial \hat { x } } { \partial \phi ^ { - } } \bigg ] . } \end{array}\tag{5}
$$

We leverage the high-quality ground x available through the simulated artifacts. This enables a score-matching term between x and xË. This formulation has been successful in image superresolution [20]. However, as discussed, the 3DGS compression artifacts are complex. To circumvent this issue, we perform restoration at $t _ { 0 }$

Perceptual Matching step, on top of the restoring distribution matching formulation, utilizes $\ell _ { 2 }$ and lpips loss functions to minimize the perceptual loss. The final optimization objective of $\phi ^ { - }$ becomes

$$
{ \mathcal L } _ { \phi ^ { - } } = { \mathcal L } _ { K L } + \ell _ { 2 } ( x , \hat { x } ) + l p i p s ( x , \hat { x } ) .\tag{6}
$$

During inference, all steps except artifact restoration are omitted. In other words, for an image ËI and with latents ${ \tilde { x } } =$ $\mathcal { E } ( { \tilde { I } } )$ the restored latents xË are estimated as

$$
\hat { x } = \tilde { x } _ { t _ { 0 } } - \sigma _ { t _ { 0 } } \epsilon _ { \theta , \phi ^ { - } } ( \tilde { x } _ { t _ { 0 } } , t _ { 0 } )\tag{7}
$$

where $\tilde { x } _ { t _ { 0 } }$ is the forward model in Eq. 2 and $t _ { 0 }$ is a hyperparameter that specifies the position on the diffusion trajectory for restoration of the complex 3DGS artifacts, enabling high perceptual performance.

The parameters $\phi ^ { + }$ are trained to minimize the following diffusion loss term where $t \sim \mathcal { U } [ t _ { m i n } , t _ { m a x } ] , \varepsilon \sim \mathcal { N } ( 0 , 1 )$ and $\hat { x } _ { t }$ is the forward model in Eq.2 with $\hat { x } _ { T } = \varepsilon$ applied to the predicted restoration

$$
\mathcal { L } _ { \phi ^ { + } } = \ell _ { 2 } ( \varepsilon , \epsilon _ { \theta , \phi ^ { + } } ( \hat { x } _ { t } , t ) ) .\tag{8}
$$

The training oscilates between optimizing for $\mathcal { L } _ { \phi ^ { - } }$ and $\mathcal { L } _ { \phi ^ { + } }$ , updating the parameters $\phi ^ { - }$ and $\phi ^ { + }$ respectively. The first restores 3DGS compression artifacts by matching the distribution of natural images and perceptual fidelity, while the latter enables modeling the restored image distribution for distribution-matching distillation. We name our overall approach NiFi.

## 4. EXPERIMENTS

## 4.1. Implementation Details

We used the DL3DV dataset with $1 0 ^ { 3 }$ scenes to create the simulated 3DGS compression artifacts dataset [27]. We set the minimum number of primitives for pruning $c _ { m i n } = 4 0 9 6$ and selected the number of primitives at three rates as described in Sec. 3. We trained the two low-rank adapters, $\phi ^ { - }$ and $\phi ^ { + }$ with rank 64, on the diffusion backbone of Stable Diffusion 3 (SD3) [26]. We used classifier-free guidance to compute $s _ { \mathrm { r e a l } }$ and $s _ { \mathrm { r e s t o r e } }$ with a guidance scale of 7.5. The parameters $\phi ^ { - }$ and $\phi ^ { + }$ are updated one after another to minimize the objectives presented in Eq. 6 and Eq. 8, respectively. To optimize these parameters, we used AdamW with learning rates of $5 \times 1 0 ^ { - 6 }$ and $1 0 ^ { - 6 }$ , weight decay of $1 0 ^ { - 4 }$ , and gradient clipping of 1.0. We followed Dong et al. in setting $\alpha = 0 . 7$ and Difix3D in setting $t _ { 0 } ~ = ~ 1 9 9$ We selected a random training view from each scene as each element of a minibatch, rather than randomly selecting from the set of all frames; hence, an epoch consisted of $\mathrm { 1 0 ^ { \overline { { 3 } } } }$ steps. We trained for $6 0 \times 1 0 ^ { 3 }$ steps with a batch size of 4 on an NVIDIA H200 GPU. This training took approximately 2 days. As the backbone is a text-to-image model, we used Qwen2.5-VL to extract prompts from the first training frame of each scene [28]. We disabled these prompts with a probability of $\frac { 1 } { 1 0 }$ during training. In practice, we included low-rank adapter weights of the encoder E in the set $\phi ^ { - }$

## 4.2. Evaluation

We used three evaluation datasets: Mip-NeRF360 [23], Tanks & Temples [24], and DeepBlending [25]. We utilized the state-of-the-art 3DGS compression method, HAC++, at three extremely low rates obtained through setting the rate parameter $\lambda \in \{ 0 . 1 , 0 . 5 , 1 . 0 \}$ [3]. We executed the restoration as in Eq. 7, given compressed renders from novel views. We compared our approach for restorationg for extreme-low-rate 3DGS compression, NiFi, with baselines of classical restoration: BM3D [6], deep learning-based restoration: SwinIR [7], diffusion-based restoration: DiffBIR [8], and 3DGS restoration: Difix3D [9]. As an ablation study, we report results without $t _ { 0 }$ and utilize the common one-step-diffusion-model practice by setting $t _ { 0 } = T , \mathrm { i . e . , } \hat { x } = \tilde { x } - \sigma _ { T } \epsilon _ { \theta , \phi ^ { - } } ( \tilde { x } , T )$ . We furthermore report a training-free approach, Img2Img, that utilizes SD3 to denoise from $\tilde { x } _ { t _ { 0 } }$ . As we do not aim for pixelto-pixel fidelity but for high perceptual quality, we report two perceptual quality metrics: LPIPS and DISTS.

<!-- image-->  
Fig. 3: Qualitative results of bicycle, kitchen, truck, and playroom scenes. Backgrounds are compressed 3DGS renders with overlayed restoration results compared within the highlighted areas . The additional highlighted area in the bicycle scene shows the overemphasized high-frequency components that are introduced by our method.

## 4.3. Results & Discussion

We present the quantitative results, along with the baseline uncompressed 3DGS-30K rate and perceptual performance, in Tab. 1. For DeepBlending, at comparable perceptual quality to 3DGS-30K, our method achieves a rate reduction from 555 MB to 0.599 MB, corresponding to a 927Ã compression. Furthermore, the results show that our method significantly improves perceptual performance at lower rates, extending the unrestored HAC++ baselineâs operation to extremely low rates. Furthermore, we outperform the image restoration baselines, confirming that our approach is more effective against 3DGS compression artifacts, and a 3DGS restoration baseline. Finally, intermediate mapping to $t _ { 0 }$ significantly improves performance, as shown in the ablation study. These results show that, combined with HAC++, our approach enables high perceptual performance at extreme rates.

We further present qualitative results in Fig. 3. These qualitative results show that our method preserves scene details with perceptual quality superior to that of the baselines. However, we would also like to point out that the restoration artifacts that arise from overemphasizing high-frequency regions. We observe this phenomenon especially in finegrained regions, such as the highlighted grassy area in the bicycle scene shown in Fig. 3.

## 5. CONCLUSION

We introduced NiFi, an extreme 3DGS compression method that extends variational diffusion distillation for restoring 3DGS compression artifacts, enabling 3DGS compression at extremely low rates, reaching 0.110 MB. We also demonstrated that mapping to an immediate point on the diffusion trajectory significantly improves perceptual performance. Our method achieved state-of-the-art perceptual performance compared to the baselines at such low rates and approached a 1000Ã rate improvement over an uncompressed 3DGS. One limitation of our work is the overemphasis on high-frequency regions, especially in areas with many details, such as the grass in the bicycle scene in Fig. 3. Our future work will focus on explaining and alleviating this limitation.

## 6. REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, andÂ¨ George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â 1, 2023.

[2] Muhammad Salman Ali, Chaoning Zhang, Marco Cagnazzo, Giuseppe Valenzise, Enzo Tartaglione, and Sung-Ho Bae, âCompression in 3d gaussian splatting: A survey of methods, trends, and future directions,â arXiv preprint arXiv:2502.19457, 2025.

[3] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai, âHac++: Towards 100x compression of 3d gaussian splatting,â IEEE TPAMI, 2025.

[4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, âHigh-resolution image synthesisÂ¨ with latent diffusion models,â in CVPR, 2022, pp. 10684â 10695.

[5] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht- Â¨ man, Fredo Durand, William T Freeman, and Taesung Park, âOne-step diffusion with distribution matching distillation,â in CVPR, 2024, pp. 6613â6623.

[6] Aram Danielyan, Vladimir Katkovnik, and Karen Egiazarian, âBm3d frames and variational image deblurring,â IEEE TIP, vol. 21, no. 4, pp. 1715â1728, 2011.

[7] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte, âSwinir: Image restoration using swin transformer,â in ICCV, 2021, pp. 1833â1844.

[8] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong, âDiffbir: Toward blind image restoration with generative diffusion prior,â in ECCV. Springer, 2024, pp. 430â448.

[9] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling, âDifix3d+: Improving 3d reconstructions with single-step diffusion models,â in CVPR, 2025, pp. 26024â 26035.

[10] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and Hamed Pirsiavash, âCompgs: Smaller and faster gaussian splatting with vector quantization,â in ECCV. Springer, 2024, pp. 330â349.

[11] Wenkai Liu, Tao Guan, Bin Zhu, Luoyuan Xu, Zikai Song, Dan Li, Yuesong Wang, and Wei Yang, âEfficientgs: Streamlining gaussian splatting for large-scale high-resolution scene representation,â IEEE MultiMedia, 2025.

[12] Francesco Di Sario, Riccardo Renzulli, Marco Grangetto, Akihiro Sugimoto, and Enzo Tartaglione, âGode: Gaussians on demand for progressive level of detail and scalable compression,â arXiv preprint arXiv:2501.13558, 2025.

[13] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al., âLightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,â NeurIPS, vol. 37, pp. 140138â140158, 2024.

[14] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava, âEagles: Efficient accelerated 3d gaussians with lightweight encodings,â in ECCV. Springer, 2024, pp. 54â71.

[15] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in CVPR, 2024, pp. 20654â 20664.

[16] Lei Liu, Zhenghao Chen, Wei Jiang, Wei Wang, and Dong Xu, âHemgs: A hybrid entropy model for 3d gaussian splatting data compression,â arXiv preprint arXiv:2411.18473, 2024.

[17] Soonbin Lee, Fangwen Shu, Yago Sanchez, Thomas Schierl, and Cornelius Hellge, âCompression of 3d gaussian splatting with optimized feature planes and standard video codecs,â in ICCV, October 2025, pp. 25496â25505.

[18] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan, âReal-esrgan: Training real-world blind super-resolution with pure synthetic data,â in ICCV, 2021, pp. 1905â1914.

[19] Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, and Zhibo Chen, âDiffusion models for image restoration and enhancement: a comprehensive survey,â IJCV, vol. 133, no. 11, pp. 8078â8108, 2025.

[20] Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou, âTsdsr: One-step diffusion with target score distillation for realworld image super-resolution,â in CVPR, 2025, pp. 23174â 23184.

[21] Jiaxin Wei, Stefan Leutenegger, and Simon Schaefer, âGsfix3d: Diffusion-guided repair of novel views in gaussian splatting,â arXiv preprint arXiv:2508.14717, 2025.

[22] Seungjoo Shin, Jaesik Park, and Sunghyun Cho, âLeveraging learned image prior for 3d gaussian compression,â in ICCV, 2025, pp. 3047â3056.

[23] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman, âMip-nerf 360: Unbounded antialiased neural radiance fields,â in CVPR, 2022, pp. 5470â5479.

[24] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Trans. on Graph., vol. 36, no. 4, pp. 1â13, 2017.

[25] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Trans. on Graph., vol. 37, no. 6, pp. 1â15, 2018.

[26] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Â¨ Axel Sauer, Frederic Boesel, et al., âScaling rectified flow transformers for high-resolution image synthesis,â in ICML, 2024.

[27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al., âDl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision,â in CVPR, 2024, pp. 22160â22169.

[28] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin, âQwen2.5-vl technical report,â arXiv preprint arXiv:2502.13923, 2025.