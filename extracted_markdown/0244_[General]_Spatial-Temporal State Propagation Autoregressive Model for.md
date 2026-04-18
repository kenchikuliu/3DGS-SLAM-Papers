# Spatial-Temporal State Propagation Autoregressive Model for 4D Object Generation

Liying Yang1â, Jialun Liu2â , Jiakui Hu3, Chenhao Guan1, Haibin Huang2, Fangqiu Yi2, Chi Zhang2, Yanyan Liang1â 

1 Macau University of Science and Technology 2 TeleAI 3 Peking University â  Corresponding Authors

## Abstract

Generating high-quality 4D objects with spatial-temporal consistency is still formidable. Existing diffusion-based methods often struggle with spatial-temporal inconsistency, as they fail to leverage outputs from all previous timesteps to guide the generation at the current timestep. Therefore, we propose a Spatial-Temporal State Propagation AutoRegressive Model (4DSTAR), which generates 4D objects maintaining temporal-spatial consistency. 4DSTAR formulates the generation problem as the prediction of tokens that represent the 4D object. It consists of two key components: (1) The dynamic spatial-temporal state propagation autoregressive model (STAR) is proposed, which achieves spatial-temporal consistent generation. Unlike standard autoregressive models, STAR divides prediction tokens into groups based on timesteps. It models long-term dependencies by propagating spatial-temporal states from previous groups and utilizes these dependencies to guide generation at the next timestep. To this end, a spatialtemporal container is proposed, which dynamically updating the effective spatial-temporal state features from all historical groups, then updated features serve as conditional features to guide the prediction of the next token group. (2) The 4D VQ-VAE is proposed, which implicitly encodes the 4D structure into discrete space and decodes the discrete tokens predicted by STAR into temporally coherent 4D Gaussians. Experiments demonstrate that 4DSTAR generates spatial-temporal consistent 4D objects, and achieves performance competitive with diffusion models.

## 1. Introduction

Generating 4D object has made tremendous progress. The recent 4D object generation approaches can be called into two main categories. Optimization methods [1, 12, 39, 40, 50, 56, 58, 60] generate 4D objects by score distilling [30] prior knowledge from pre-trained diffusion models. But the score distillation is sensitive to prompts [34], which limit the application of optimization methods.

<!-- image-->  
Figure 1. Diffusion-based methods, such as previous work [57], fail to leverage outputs from all previous timesteps to guide the generation at the current timestep, which generates results with inconsistent appearance at some timesteps. Our 4DSTAR alleviates this issue by leveraging historical outputs to guide the generation at the current timestep.

Feed-forward methods [18, 34, 53, 57, 62, 63] directly train a diffusion model on 4D datasets. However, they often produce 4D objects with spatial-temporal inconsistency, as they struggle to utilize the outputs from all previous timesteps to guide the generation at the current timestep. In an extreme case (Figure 1), when generating results over a long time span, previous work only relies on the input video and limited view information. This information fails to adequately assist the model in inferring results that are temporally coherent between timestep 1 and timestep 24.

To address these limitations, we present a novel feed-forward Spatial-Temporal State Propagation AutoRegressive Model, named 4DSTAR, which generates temporal-spatial consistency 4D objects. 4DSTAR formulates the 4D generation as the token prediction. It comprises a dynamic spatial-temporal state propagation auto-regressive model (STAR) and a 4D VQ-VAE.

Specifically, we propose dynamic spatial-temporal state propagation autoregressive model (STAR), which models long-term dependencies across generations from previous timesteps via spatial-temporal state propagation, and leverages dependencies to guide next timestep generation. Firstly, it is necessary for the model to accurately observe all changes during the motion of a 4D object, because changes in some areas at a future timestep can refer to corresponding areas at previous timesteps. An intuitive analogy is to pause the object at any timestep and observe it from all angles. Inspired by this concept, STAR divides prediction tokens into multiple groups based on timesteps.

Then, to model long-term dependencies, we propose a Spatial-Temporal Container (S-T Container) integrated into STAR. It is designed to adaptively update effective spatialtemporal state features from all historical prediction groups. In particular, we assume some token features across different historical groups share similarities in texture and geometry. After merging these similar features, the remaining features constitute the effective spatial-temporal state, which provides the reference context for predicting the next token group. Based on this, the S-T container retrieves tokens from all historical groups and dynamically updates the set of merged features. These updated features then serve as conditional features guiding the prediction of the next token group and are sequentially updated and propagated throughout the autoregressive prediction process of STAR. In other words, under T frames, when predicting results at any given t (t â¤ T ), we leverage existing 1 to t â 1 historical groupsâ spatial-temporal state features to predict results at timestep t, which enables long-term dependencies across 1 to t results prediction.

To decode discrete tokens generated by STAR, we propose a 4D VQ-VAE that implicitly encodes the 4D structure into discrete space and decodes the token sequences into dynamic 3D Gaussians. To preserve temporal stability, our 4D VQ-VAE avoids compression along the temporal axis. The Spatial-Temporal Decoder within 4D VQ-VAE consists of a Static GS Generation and Spatial-Temporal Offset Predictor (STOP). Static GS generation decodes discrete tokens to static Gaussian features. At the same time, STOP jointly leverages cross-frame temporal information from token sequences and static Gaussian features to decode per-timestep Gaussian offsets, thereby establishing explicit point-level correspondence across frames. This enables spatiotemporal fusion to correct static Gaussians into a canonical 4D space at each timestep. Finally, 4D VQ-VAE produces spatialtemporal consistent dynamic 3D Gaussians.

The contributions can be summarized as follows:

â¢ To the best of our knowledge, we are the first to propose

an autoregressive model for 4D object generation.

â¢ To enforce spatial-temporal consistent generation, we propose a Dynamic Spatial-Temporal State Propagation Autoregressive Model (STAR). It models long-term dependencies across previous predictions via spatialtemporal state propagation to guide the generation at the current timestep.

â¢ To decode the tokens predicted by STAR, we propose a 4D VQ-VAE, which implicitly encodes the 4D structure into discrete space and decodes discrete tokens into temporally coherent dynamic 3D Gaussians.

â¢ Experimental results demonstrate that our method can generate spatial-temporal consistent 4D objects and achieves performance competitive with diffusion models.

## 2. Related Works

## 2.1. 3D Generation

Generating high-quality 3D representations, such as meshes [19, 22, 49, 55], NeRFs [9, 26, 27], and 3D Gaussian splats [15, 42, 65] has become a rapidly evolving research area. Early methods [32, 36, 43, 54] distill 2D diffusion priors [29, 35] into 3D via Score Distillation Sampling (SDS) [31], enabling text- or image-to-3D synthesis but suffering from inefficiency and view inconsistency caused by per-instance optimization and single-view ambiguity [22]. To improve scalability, later works [20â22, 37] decouple the process into multi-view generation followed by 3D reconstruction, often fine-tuning pretrained 2D diffusion models on large 3D datasets [5, 6]. Although these approaches produce visually appealing results, the reconstructed meshes are often unsuitable for downstream tasks such as animation. More recent methods [23, 44, 61, 64] abandon 2D priors and train dedicated 3D generative models from scratch, achieving more accurate and detailed geometry.

## 2.2. 4D Generation

Extending 3D generation into the spatiotemporal domain, 4D generation aims to synthesize dynamic 3D content from text, images, videos, or static assets, requiring consistent geometry and temporally coherent motion. MAV3D [39] applies score distillation sampling (SDS) from video diffusion models to optimize dynamic NeRFs, while 4D-FY [1] combines image-, video-, and 3D-aware supervision for better structural fidelity. Consistent4D [12], STAG4D [60], SC4D [50], DreamMesh4D [17] and DS4D [56] enhance temporal consistency via frame interpolation and multiview fusion but remain limited by slow optimization and color oversaturation from SDS [30]. Recent models such as Diffusion4D [18], 4Diffusion [63], L4GM [34], GVFDiffusion [62], and SV4D [52, 57] employ 4D-aware diffusion to generate orbital views of dynamic assets; however, they struggle with spatial-temporal inconsistency. In contrast, our method solves this by utilizing outputs from all previous timesteps to guide the generation at current timestep.

## 2.3. Autoregressive Visual Generation

Autoregressive (AR) modeling, pioneered by Pixel-CNN [47], formulates image generation as a sequential prediction problem over pixels. Subsequent works such as $\mathrm { v Q } .$ VAE [48] and VQGAN [8] extend this paradigm by quantizing image patches into discrete tokens and training transformers to learn AR priors, analogous to language modeling [3]. To enhance reconstruction fidelity, RQVAE [16] introduces multi-scale quantization, while VAR [45] reformulates AR modeling as next-scale prediction, markedly improving sampling efficiency. Recent efforts further scale AR transformers for text-conditioned visual generation at large scale [10]. In the 3D domain, early studies such as PolyGen [28] and MeshGPT [38] adopt AR models to directly generate mesh vertices and faces. However, these methods are confined to geometric representations and face challenges in scaling to diverse 3D object datasets [5]. In contrast, we propose an autoregressive model for 4D object generation, which updates effective spatial-temporal state features provided by all historical groups, enabling temporally coherent 4D generation.

## 3. Methodology

We aim to generate high-quality 4D objects with spatialtemporal consistency. Figure. 2 shows the overview of 4DSTAR, including 4D VQ-VAE and dynamic Spatial-Temporal State Propagation auto-Regressive model.

## 3.1. 4D Vector Quantized Variational Autoencoder

In our 4D VQ-VAE, each 4D object is regarded as the spatial-temporal matrix, which is arranged by 2D view images $x \in \dot { \mathbb R ^ { T \times V \times H \times W \times 3 } }$ , where T denotes the number of frames, V denotes the number of views. To capture both fine-grained details and high-level semantics within each 2D view images, we employ the encoder of UniTok [25]. The encoder encodes the spatial-temporal matrix to latent vector, and then obtain the corresponding discrete tokens. Specifically, the matrix is quantized into $\overline { { q } } \in Q ^ { T \times V \times n \times \frac { d } { n } }$ where n is the dimension of latent vector, n denotes the number of chunks which used to split latent vector.

The discrete tokens extracted from the matrix require a decoder for reconstruction. Although a straightforward solution is to employ the UniTok decoder, it suffers from a fundamental limitation due to the gap between 2D and 4D representations. Specifically, UniTok decoder is designed for independent 2D image reconstruction and fails to capture the inherent geometric constraints in 4D. Consequently, this leads to spatial-temporal inconsistencies across the reconstructed frames. To address this issue, we propose a spatial-temporal decoder (STD). In contrast to UniTok decoder, STD decodes tokens as dynamic 3D Gaussians.

The STD commits to reconstruct 4D results represented by 4D Gaussian splatting. Specifically, for input discrete tokens q, we employ the multi-head attention modules for token factorization projection. It converts the discrete tokens into continuous tokens $S ~ \in ~ \mathbb { R } ^ { T \times V \times C \times N }$ , where C denotes the dimension of tokens, N denotes the number of tokens. Then, through encoding and decoding these continuous tokens, the coarse decoder in STD can directly decode coarse Gaussians $G ^ { t } ~ = ~ \{ g _ { 1 } ^ { t } , g _ { 2 } ^ { t } , . ~ . ~ . , g _ { n } ^ { t } \}$ , with the parameters $g _ { i } ^ { t } ~ = ~ \{ x _ { i } ^ { t } , s _ { i } ^ { t } , \gamma _ { i } ^ { t } , \alpha _ { i } ^ { t } , c _ { i } ^ { t } \}$ for each timestep t. To further correct these static Gaussians into a canonical 4D space at each timestep, we propose Spatial-Temporal Offset Predictor (STOP) inserted into STD, which jointly leverages cross-frame temporal information from token sequences and static GS features.

Spatial-Temporal Offset Predictor (STOP). Specifically, we input static Gaussians and continuous tokens t of each object into the STOP. For the dynamic 3D Gaussians $\mathbf { G } \ = \ \{ G ^ { 1 } , G ^ { 2 } , \ldots , G ^ { T } \}$ in the whole timesteps T , and continuous tokens $\mathbf { S } _ { v } = \left\{ S _ { v } ^ { 1 } , S _ { v } ^ { 2 } , \ldots , S _ { v } ^ { T } \right\}$ belonging to the whole timesteps T at each view v, we calculate the cross-attention among static Gaussian features and continuous tokens along timestep axes. In detail, we set the G as query and $\mathbf { S } _ { v }$ as key and value. During the cross-attention calculation, each static Gaussian features can aggregate the global temporal information based on the similar geometry and texture relationships among continuous tokens in temporal axes, thereby leveraging temporal context to obtain the coarse offset features at each timestep. Then, a 3D-Unet module is used to refine the coarse offset features, which enhance the 3D awareness of the features. The refined offset features $f _ { o } ^ { t } = \{ x _ { i o } ^ { t } , s _ { i o } ^ { t } , \gamma _ { i o } ^ { t } , \alpha _ { i o } ^ { t } , c _ { i o } ^ { t } \}$ can be split as the offset value of each parameter of the dynamic 3D Gaussians. At last, we update the parameters of dynamic 3D Gaussians as $\hat { g _ { i } } ^ { t } = \{ x _ { i } ^ { t } + x _ { i o } ^ { t } , s _ { i } ^ { t } + s _ { i o } ^ { t } , \gamma _ { i } ^ { t } + \gamma _ { i o } ^ { t } , \alpha _ { i } ^ { t } + \alpha _ { i o } ^ { t } , c _ { i } ^ { t } + c _ { i o } ^ { t } \}$

Loss Function. We employ pixel-level rendering loss $\mathcal { L } _ { R }$ among ground truth and rendering views, and use a discriminator loss $\mathcal { L } _ { G }$ to enhance reconstruction fidelity [14]. Furthermore, we introduce the optical flow loss $\mathcal { L } _ { F }$ to guide the motion modeling. For 4D VQ-VAE training, the overall loss function $\mathcal { L } _ { \mathrm { V A E } } = \alpha \mathcal { L } _ { R } + \beta \mathcal { L } _ { G } + \gamma \mathcal { L } _ { F }$ , where $\alpha , \beta , \gamma$ is weight. More details see supplementary materials.

## 3.2. Dynamic Spatial-Temporal State Propagation Auto-Regressive Model (STAR)

In Sec. 3.1, we introduce how our 4D VQ-VAE encodes and decodes the tokens representing 4D objects. In this section, we introduce our STAR how to predict these tokens.

To build long-term dependencies, a simple approach is to use a standard auto-regressive model (e.g., LlamaGen [41]) to predict these tokens. However, this approach faces significant challenges. The primary difficulty is the need to precisely predict a massive sequence of tokens (over 40,000) that encapsulates a 4D object. This challenge is compounded by high token density, which complicates accurate prediction, and the fact that not all historical information is useful for forecasting the next token. To address these limitations, we propose a Dynamic Spatial-Temporal State Propagation Auto-Regressive Model (STAR), which updates effective spatial-temporal state features based on historical groups. In this section, we provide a detailed introduction to STAR.

<!-- image-->  
Figure 2. The overall pipeline of our 4DSTAR. 4DSTAR consists of two key components: (a) 4D VQ-VAE. Given a 4D object, we first render it as a spatial-temporal matrix. Then the matrix is encoded by Encoder, and is compressed into discrete tokens. Static GS Generation decodes these tokens to static Gaussians. Meanwhile, Spatial-Temporal Offset Predictor (STOP) corrects static Gaussians into a canonical 4D space at each timestep. Finally, the model outputs dynamic 3D Gaussians. (b) Dynamic Spatial-Temporal State Propagation Autoregressive Model (STAR). The text and video conditions, which are compressed by an image tokenizer are concatenated before the start token as the context. The conditions can either expect the model to generate tokens. The SEP signals the model to begin generating tokens. Then, camera pose and timestep conditions are integrated. When the model starts to predict the next group, the historical groups are integrated into Spatial-Temporal Container (S-T Container). S-T Container updates effective spatial-temporal state features. The features serve as conditional features to guide the prediction of the next token group. Finally, STAR predicts all tokens that represent a 4D object.

## 3.2.1. Conditions

Firstly, we introduce the conditions integrated into STAR.

Text Condition. To extract text features, we utilize FLAN-T5 XL [4] as the text encoder. Then, the text features are projected by a text tokenizer, which contains an MLP, and serve as the prefill token embeddings in STAR.

Camera Condition. To condition the camera pose within a group, following [11, 13], we apply the Plucker Â¨ Embedding. This embedding encodes the origin and direction of the ray at each spatial location. The ray is then incorporated into STAR as the Shift Positional Encoding (SPE).

Timestep Condition. To condition the temporal information across groups, we design a Timestep encoder that maps each Timestep t to a temporal embedding $\boldsymbol { \hat { t } } \in \mathbb { R } ^ { D _ { p } }$ p , with $D _ { p }$ aligning the dimensions of the Plucker Embedding. Â¨ Similar to to the latter, this temporal embedding is incorporated into STAR as the SPE.

Monocular Video Condition. Given a monocular video of T frames, we represent it as the set ${ \textbf { F } } =$ $\{ F _ { 1 } , F _ { 2 } , \ldots , F _ { T } \}$ . The 4D VQ-VAE encoder first converts this input into discrete tokens. These tokens are then projected by an image tokenizer, which contains an MLP, to form the prefill token embeddings for STAR.

## 3.2.2. Network Architecture

Based on the conditions in Sec. 3.2.1, STAR is committed to predicting the discrete tokens that represent a 4D object.

Dividing Groups. First, building upon the concept introduced in Sec. 1, STAR divides the prediction tokens into multiple groups, as illustrated in Figure 2 (b). Specifically, we consider a prediction token sequence Q = $\{ q _ { 1 } ^ { 1 } , q _ { 1 } ^ { 2 } , q _ { 1 } ^ { 3 } , \ldots , q _ { t } ^ { v } , \ldots , q _ { T } ^ { V } \}$ , where T denotes the number of timesteps and V denotes the number of views. This sequence is partitioned into $T$ groups, denoted as $\mathrm { ~ \bf ~ L ~ } =$ $\{ L _ { 1 } , L _ { 2 } , \ldots , L _ { T } \}$ . The group $L _ { t }$ at time t, contains the partial prediction tokens $\overset { \vartriangle } { \boldsymbol { L } _ { t } } = \bar { \{ q _ { t } ^ { 1 } , q _ { t } ^ { 2 } , \ldots , q _ { t } ^ { V } \} }$ . Thus, all prediction tokens are organized into groups, which STAR then predict them sequentially. During training, the groups L are projected by the image tokenizer into token features LË = $\{ \hat { L } _ { 1 } , \hat { L } _ { 2 } , \dots , \hat { L } _ { T } \}$ . Then the token features and conditions serve as the input to STAR. During inference, STAR autoregressively predicts the token sequences for each group based on the conditioning inputs.

Spatial-Temporal Container. We propose the Spatial-Temporal Container (S-T Container) to build long-term dependencies by dynamically updating spatial-temporal state features from historical groups. S-T Container is designed to merge token features that exhibit similar texture and geometry across historical groups. To identify and extract these potentially similar token features, we employ a knearest neighbor based density peaks clustering algorithm (DPC-KNN) [7].

Specifically, given token features $\hat { \mathbf { L } } _ { t \in [ 1 , t - 1 ] }$ from group 1 to group t â 1 (i.e., time 1 to time $t - 1 )$ , we compute the local density $\rho _ { i }$ of each token within groups:

$$
\rho _ { i } ^ { m } = \mathsf { e x p } ( - \frac { 1 } { K } \sum _ { y _ { j } ^ { n } \in \mathsf { K N N } ( y _ { i } ^ { m } ) } \| y _ { i } ^ { m } - y _ { j } ^ { n } \| _ { 2 } ^ { 2 } ) ,\tag{1}
$$

where $\mathtt { K N N } ( y _ { i } ^ { m } )$ denotes the k-nearest neighbors of the m-th token feature in i-th group. Next, we compute the similar score $\varpi$ as the minimal distance among token features with higher local density:

$$
\varpi _ { i } ^ { m } = \left\{ \begin{array} { c l } { \underset { { j , n : \rho _ { j } ^ { n } > \rho _ { i } ^ { m } } } { m i n } \Vert y _ { i } ^ { m } - y _ { j } ^ { n } \Vert _ { 2 } ^ { 2 } , } & { \mathrm { ~ i f ~ } \exists j , n \mathrm { ~ s . t ~ } \rho _ { j } ^ { n } > \rho _ { i } ^ { m } . } \\ { \underset { { j , n } } { m a x } \Vert y _ { i } ^ { m } - y _ { j } ^ { n } \Vert _ { 2 } ^ { 2 } , } & { \mathrm { ~ o t h e r w i s e . } } \end{array} \right.\tag{2}
$$

where $\rho _ { i } ^ { m }$ denotes local density. Then we set cluster centers based on the score $\rho _ { i } ^ { m } \times \varpi _ { i } ^ { m }$ . A higher score means a higher possibility of being a cluster center.

Inspired by feature fusion [33, 51, 59], we use an MLP to predict the dissimilar score Ï of each token feature. Then we merge the token features according to $\sigma ,$ the merged token features $\begin{array} { r } { \hat { y } _ { i } = \sum _ { j \in C _ { i } } \sigma _ { j } y _ { j } } \end{array}$ , where $C _ { i }$ denotes the set of the i-th cluster, and $y _ { j }$ means the j-th token features in the i-th cluster. Next, we refine the merged token features via multi-head attention, obtaining the updated features $\hat { y } _ { i }$

The updated features $\hat { y } _ { i }$ represents the effective spatialtemporal state features provided by groups 1 to t â 1. These features then serve as condition features, integrated into STAR via MLPs, for predicting the tokens in group t. As the prediction progresses through the groups, the S-T container iteratively updates the refined token features. In other words, it gradually propagates the dynamic spatialtemporal state, thereby building long-term dependencies across all timesteps.

Transformer Architecture. We employ Transformer for auto-regressive modeling, which is consisted with standard auto-regressive model. The Transformer is developed based on Llama [46].

Loss Function. Following auto-regressive models [41, 46], our STAR generates the conditional probability $p ( q _ { s } | q _ { < s } )$ of token $q _ { s }$ at each position s. Then, the crossentropy (CE) loss is defined as the average of the negative log-likelihoods over all vocabulary positions:

$$
\mathcal { L } _ { a r } = - \frac { 1 } { S } \sum _ { s = 1 } ^ { S } 1 \circ \mathbf { g } p ( q _ { s } | q _ { < t } ) ,\tag{3}
$$

The Eq. 3 indicates that STAR studies the transformation process from previous s â 1 tokens to s-th tokens. Due to we have $T$ groups in STAR, we employ chunked CE loss, which we split the chunk based on the group. Each chunk includes one group, formally,

$$
\mathcal { L } _ { A R } = \sum _ { g = 1 } ^ { G } \mathsf { C h u n k } ( L _ { a r } ^ { 1 } , L _ { a r } ^ { 2 } , \ldots , L _ { a r } ^ { G } ) .\tag{4}
$$

where G means the number of groups.

## 4. Experiments

## 4.1. Implementation Details

Training Dataset and Metrics. We train 4D VQ-VAE and STAR on the train set of Objaverse [5] and Objaverse-XL [6], which includes 56K 4D objects. For text prompts, we first utilize those provided by Cap3D [24]. However, as these prompts describe static appearance, we employ Qwen [2] to generate additional prompts that capture object motion. To evaluate the quality of our 4D generation, we adopt metrics from [12, 60], including CLIP, LPIPS, FVD, and FID-VID. The same metrics are also used to evaluate the reconstruction quality of 4D VQ-VAE. All evaluations are conducted on renderings at a resolution of 512 Ã 512.

Training. We train our 4D VQ-VAE on 8 H100 GPUs and STAR on 16 H100 GPUs. More details can be found in the Supplementary Materials.

## 4.2. Main Results

## 4.2.1. 4D Object Reconstruction

Evaluation Dataset. To evaluate the performance on 4D object reconstruction, we construct a test set of 100 objects from Objaverse and Objaverse-XL. We render 8 views of each timestep for each object. These views are grouped as a spatial-temporal matrix as the input of each method. Besides, we select 4 views (front, back, left, and right) of each timestep for each object, which are used as ground truth.

Quantitative comparisons. We compare our 4D VQ-VAE with two 2D image VQ-VAE, including VQ-VAE [41] and UniTok [25]. As shown in Table. 3, our 4D VQ-VAE outperforms other VQ-VAE across all quality metrics. It demonstrates that our VQ-VAE has both superior reconstruction fidelity and better temporal coherence. This is attributed to our 4D VQ-VAE leveraging spatial-temporal information within tokens to reconstruct results.

<!-- image-->

Figure 3. Qualitative comparison with VQ-VAE [41] and UniTok [25] on 4D reconstruction. For VQ-VAE and UniTok, we employ them to reconstruct 2D view images. For our 4D VQ-VAE, we render results under corresponding views. Our 4D VQ-VAE can reconstruct the results with temporal coherence, while VQ-VAE and UniTok cannot reconstruct with temporal coherence.  
<!-- image-->  
Figure 4. Qualitative comparison with the state-of-the-art methods [34, 57, 60, 62] on video-to-4D generation. For each method, we render results under two novel views at two timesteps. Our method achieves high-quality generation with spatial-temporal consistency.

<table><tr><td>Method</td><td>CLIP â</td><td>LPIPS</td><td>FVDâ</td><td>FID-VID</td></tr><tr><td>VQ-VAE [41]</td><td>0.938</td><td>0.067</td><td>326.570</td><td>13.447</td></tr><tr><td>UniTok [25]</td><td>0.973</td><td>0.050</td><td>151.238</td><td>4.402</td></tr><tr><td>4D VQ-VAE (Ours)</td><td>0.973</td><td>0.048</td><td>133.372</td><td>4.175</td></tr></table>

Table 1. Evaluation and comparison of the performance on 4D object reconstruction. The best score is highlighted in bold. All the experiments of the methods are carried out using the code from their official GitHub repository.

Qualitative comparisons. The qualitative results on test set are presented in Figure. 3. Obviously, the reconstruction quality and temporal coherence of our 4D VQ-VAE both are better than other methods. Specifically, for the top object, VQ-VAE and UniTok cannot reconstruct the texture details of eye at two timesteps, while our 4D VQ-VAE successfully reconstruct the details. Moreover, for the below object, VQ-VAE and UniTok reconstruct significantly inconsistent texture results at different timesteps, especially for the clothing texture. In contrast, our 4D VQ-VAE reconstructs consistent texture results at both timesteps, which once again demonstrates the ability of our 4D VQ-VAE in ensuring temporal coherence in reconstruction.

<table><tr><td>Method</td><td>CLIP â</td><td>LPIPSâ</td><td>FVDâ</td><td>FID-VIDâ</td></tr><tr><td>STAG4D [60]</td><td>0.910</td><td>0.141</td><td>752.215</td><td>27.280</td></tr><tr><td>L4GM [34]</td><td>0.926</td><td>0.132</td><td>515.430</td><td>20.125</td></tr><tr><td>SV4D 2.0 [57]</td><td>0.932</td><td>0.139</td><td>497.753</td><td>19.223</td></tr><tr><td>GVFDiffusion [62]</td><td>0.931</td><td>0.138</td><td>528.336</td><td>19.064</td></tr><tr><td>4DSTAR (Ours)</td><td>0.952</td><td>0.131</td><td>464.709</td><td>15.312</td></tr></table>

Table 2. Evaluation and comparison of the performance on video-to-4D object generation. The best score is highlighted in bold. All the experiments of the methods are carried out using the code from their official GitHub repository.

## 4.2.2. Video-to-4D Object Generation

Evaluation Dataset. We construct a test set of 100 objects by combining 7 objects from Consistent4D [12] and 93 additional objects from Objaverse-XL test set. To ensure fair comparisons, our model employs a neutral text prompt, âGenerate object of the following < imgs >â, which intentionally omits any specific appearance or motion descriptions. For previous works, we follow their original papers and official implementations, using their semantic text embeddings (if they have) instead of neutral text prompt.

Quantitative comparisons. We compare our model with the SOTA methods on the test set, including STAG4D [60], L4GM [34], GVFDiffusion [62], and SV4D 2.0 [57]. The quantitative results are shown in Table. 2. Our 4DSTAR consistently outperforms other methods in all metrics. Specifically, our 4DSTAR notably exceeds the other methods in FVD and FID-VID, which indicates the results generated by our 4DSTAR have fewer temporal artifacts and better temporal coherence than others. Furthermore, our 4DSTAR significantly outperforms other methods in CLIP. It demonstrates our methods are superior in terms of quality, fidelity, and spatial-temporal consistency of generation results. In summary, these improvements are attributed to our method, which models long-term dependencies across outputs from previous timesteps to enhance spatial-temporal consistent generation.

Qualitative comparisons. The qualitative results are presented in Figure. 4. Visibly, the generation fidelity and spatial-temporal consistency are both better than other methods. Specifically, when generating the results in two timesteps, the results generated by STAG4D, L4GM, and SV4D 2.0 have different degrees of blurriness and inconsistent appearance in the details, especially in areas with complex topology (e.g., boyâs hair). Moreover, when generating the results that contain large motion at two timesteps, the results generated by L4GM, GVFDiffusion, and SV4D 2.0 show inconsistent appearance, temporal incoherence, and some noisy points within motion parts. For example, for the arms of the bear, these methods generate clear textures in previous timesteps, while they generate low-quality textures in the future timestep. These methods fail to leverage outputs from previous timesteps to guide generation at the current timestep. In contrast, our 4DSTAR alleviates these issues by leveraging long-term dependencies to guide the generation, achieving high-quality generation.

## 4.3. Ablation Study

Ablation of 4D VQ-VAE. To validate the effect of STOP in 4D VQ-VAE, we conduct an experiment about whether to use STOP in model. The results are shown in Table. 3. Our method significantly outperforms the model without using STOP in all metrics, especially FVD and FID-VID. Besides, Figure. 5 shows the reconstruction results by different models at three timesteps. The model without STOP cannot reconstruct consistent texture details across timesteps, while our model, which uses STOP, accurately recovers texture details at different timesteps. It indicates the STOP leverage of spatial-temporal information among tokens to correct static Gaussians into a canonical 4D space for better temporally coherent reconstruction.

<table><tr><td>Method</td><td>CLIP â</td><td>LPIPS â</td><td>FVDâ</td><td>FID-VID</td></tr><tr><td>w/o STOP</td><td>0.968</td><td>0.057</td><td>174.757</td><td>5.575</td></tr><tr><td>4D VQ-VAE (Ours)</td><td>0.973</td><td>0.048</td><td>133.372</td><td>4.175</td></tr></table>

Table 3. Ablation Experiments of 4D VQ-VAE on the test set. Each setup is based on a modification of the immediately preceding setups. The best score is highlighted in bold.

<!-- image-->  
Figure 5. Ablation of 4D VQ-VAE. Our model which uses STOP, accurately recovers texture details at different timesteps.

Ablation of STAR. To validate the effect of S-T Container in STAR, we conduct several experiments, as shown in Table. 4. Among them, model A based on L4GM with explicit state propagation. Baseline is the naive autoregressive model. Model B, based on baseline, performs average pooling of historical groups along temporal to merge historical groups, while model C uses MLP to merge them. Our model with S-T Container significantly outperforms other models in all metrics. Our model leverages clustering of S-T Container to preserve diverse spatial-temporal information within historical token features. This process provides a crucial inductive bias for modeling long-term dependencies. However, model B and C struggle to accurately extract such information. Although model A explicitly propagates previous results to assist prediction at the next timestep, it only uses the results of one historical frame to guide the generation at the next timestep. This not only makes the model easily forget valid information from all historical results but also fails to filter useful information from historical results, ultimately leading to the generation of spatial-temporal inconsistencies that affect time coherence. In contrast, our model can consistently produce temporally consistent results. By merging regionally similar textures and topological structures via S-T container and propagating remaining features to constitute the effective spatial-temporal state across timesteps, our model gradually establishes long-term dependencies, thereby improving temporal consistency in generation. For model details and more visualizations, please see supplementary materials.

<table><tr><td>Method</td><td>CLIP â</td><td>LPIPS â</td><td>FVDâ</td><td>FID-VID</td></tr><tr><td>A: L4GM w/ Explicit State Propagation</td><td>0.930</td><td>0.132</td><td>508.192</td><td>19.714</td></tr><tr><td>Baseline</td><td>0.939</td><td>0.148</td><td>883.946</td><td>37.066</td></tr><tr><td>B: w/ Temporal Average Pooling</td><td>0.942</td><td>0.146</td><td>646.558</td><td>27.016</td></tr><tr><td>C: w/ Learnable Token Merging</td><td>0.945</td><td>0.141</td><td>579.288</td><td>21.449</td></tr><tr><td>Ours: w/S-T Container</td><td>0.952</td><td>0.131</td><td>464.709</td><td>15.312</td></tr></table>

Table 4. Ablation Experiments of STAR on the test set. Each setup is based on a modification of the immediately preceding setups. The best score is highlighted in bold.

<!-- image-->  
Figure 6. Ablation of STAR. Even slight motion within input, ours produces more temporally consistent results than Baseline. More visualizations see supplementary materials.

<!-- image-->  
Figure 7. Visualization of the clustering within S-T Container. The left is the original distribution of token features belonging to previous (historical) groups. The right visualizes the clustering centers after clustering. The clustering centers preserve the diversity of spatial-temporal information within token features.

Visualization of Clustering within S-T Container. The visualization validates the feature update principle in our S-T container. We first plot the distribution of original token features from previous groups using PCA and T-SNE, marking their centers (left of Figure. 7). After clustering these features via our S-T container, we visualize the cluster centers (right of Figure. 7). Initially concentrated token feature centers indicate many token features sharing similarities, while the well-separated clustering centers after clustering preserve diverse spatial-temporal information within token features. By merging similar token features, S-T container updates token features in historical features into features with informative spatial-temporal representations. This effectively models long-term dependencies to guide predictions for subsequent groups.

<!-- image-->  
(b) Our Results on Text-Image to Static 3D Object  
Figure 8. Visualization of (a) our text-video to 4D object generation and (b) our text-image to static 3D object generation. Our approach is capable of generating a 4D object with temporalspatial consistency and a 3D object with multi-view consistency.

## 5. Applications

In Sec. 4, we verify our 4DSTAR is effective at generating 4D objects according to the input video. Additionally, our 4DSTAR supports the text prompt and corresponding video as input, as shown in Figure. 8 (a). Furthermore, our 4DSTAR can generate static 3D objects maintaining multiview consistency when input text prompt and reference image, as shown in Figure. 8 (b).

## 6. Conclusion

In this paper, we propose a novel feed-forward Spatial-Temporal State Propagation Autoregressive Model named 4DSTAR, which generates temporal-spatial consistency 4D objects. Its STAR models long-term dependencies across generation results from previous timesteps via spatial-temporal state propagation, and leverages dependencies to guide generation at next timestep. A key component is the spatial-temporal container, integrated within STAR, which propagates prediction information from previous groups to enforce spatial-temporal consistency. Furthermore, 4D VQ-VAE is designed to decode the tokens predicted by STAR into temporally coherent dynamic 3D Gaussians. Overall, our method can generate high-quality 4D objects with temporal-spatial consistency.

## References

[1] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996â8006, 2024. 1, 2

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877â1901, 2020. 3

[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. Journal of Machine Learning Research, 25(70):1â53, 2024. 4

[5] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 3, 5

[6] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36, 2024. 2, 5

[7] Mingjing Du, Shifei Ding, and Hongjie Jia. Study on density peaks clustering based on k-nearest neighbors and principal component analysis. Knowledge-Based Systems, 99: 135â145, 2016. 5

[8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873â12883, 2021. 3

[9] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In International Conference on Machine Learning, pages 11808â11826. PMLR, 2023. 2

[10] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024. 3

[11] JiaKui Hu, Yuxiao Yang, Jialun Liu, Jinbo Wu, Chen Zhao,

and Yanye Lu. Auto-regressively generating multi-view consistent images. arXiv preprint arXiv:2506.18527, 2025. 4

[12] Yanqin Jiang, Li Zhang, Jin Gao, Weiming Hu, and Yao Yao. Consistent4d: Consistent 360Â° dynamic object generation from monocular video. In The Twelfth International Conference on Learning Representations, 2024. 1, 2, 5, 7

[13] Yash Kant, Aliaksandr Siarohin, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, and Igor Gilitschenski. Spad: Spatially aware multi-view diffusers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10026â10038, 2024. 4

[14] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401â4410, 2019. 3

[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 2

[16] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523â11532, 2022. 3

[17] Zhiqi Li, Yiming Chen, and Peidong Liu. Dreammesh4d: Video-to-4d generation with sparse-controlled gaussianmesh hybrid representation. Advances in Neural Information Processing Systems, 37:21377â21400, 2024. 2

[18] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024. 1, 2

[19] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36, 2024. 2

[20] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298â9309, 2023. 2

[21] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023.

[22] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970â9980, 2024. 2

[23] Zhang Longwen, Wang Ziyu, Zhang Qixuan, Qiu Qiwei, Pang Anqi, Jiang Haoran, Yang Wei, Xu Lan, and Yu Jingyi. Clay: A controllable large-scale generative model for creat-

ing high-quality 3d assets. arXiv preprint arXiv:2406.13897, 2024. 2

[24] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. Advances in Neural Information Processing Systems, 36: 75307â75337, 2023. 5

[25] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025. 3, 5, 6

[26] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8446â8455, 2023. 2

[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[28] Charlie Nash, Yaroslav Ganin, SM Ali Eslami, and Peter Battaglia. Polygen: An autoregressive generative model of 3d meshes. In International conference on machine learning, pages 7220â7229. PMLR, 2020. 3

[29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Â¨ Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2

[30] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1, 2

[31] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2

[32] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 2

[33] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937â13949, 2021. 5

[34] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828â56858, 2024. 1, 2, 6, 7

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image Â¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684â10695, 2022. 2

[36] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Anything-3d: Towards single-view anything reconstruction in the wild. arXiv preprint arXiv:2304.10261, 2023. 2

[37] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2

[38] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias NieÃner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19615â19625, 2024. 3

[39] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, and Yaniv Taigman. Text-to-4d dynamic scene generation. arXiv:2301.11280, 2023. 1, 2

[40] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023. 1

[41] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3, 5, 6

[42] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10208â 10217, 2024. 2

[43] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22819â22829, 2023. 2

[44] Tencent Hunyuan3D Team. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation, 2025. 2

[45] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839â84865, 2025. 3

[46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Â´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. \` Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 5

[47] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 3

[48] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3

[49] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024. 2

[50] Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai. Sc4d: Sparse-controlled video-to-4d generation and motion transfer. In European Conference on Computer Vision, pages 361â379. Springer, 2024. 1, 2

[51] Haozhe Xie, Hongxun Yao, Xiaoshuai Sun, Shangchen Zhou, and Shengping Zhang. Pix2vox: Context-aware 3d reconstruction from single and multi-view images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2690â2698, 2019. 5

[52] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 2

[53] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 1

[54] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, and Zhangyang Wang. Neurallift-360: Lifting an in-the-wild 2d photo to a 3d object with 360deg views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4479â4489, 2023. 2

[55] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024. 2

[56] Liying Yang, Chen Liu, Zhenwei Zhu, Ajian Liu, Hui Ma, Jian Nong, and Yanyan Liang. Not all frame features are equal: Video-to-4d generation via decoupling dynamic-static features. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7494â7504, 2025. 1, 2

[57] Chun-Han Yao, Yiming Xie, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation. arXiv preprint arXiv:2503.16396, 2025. 1, 2, 6, 7

[58] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 1

[59] Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang. Not all tokens are equal: Human-centric visual analysis via token clustering transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11101â 11111, 2022. 5

[60] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision, pages 163â179. Springer, 2025. 1, 2, 5, 6, 7

[61] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 42(4):1â16, 2023. 2

[62] Bowen Zhang, Sicheng Xu, Chuxin Wang, Jiaolong Yang, Feng Zhao, Dong Chen, and Baining Guo. Gaussian variation field diffusion for high-fidelity video-to-4d synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12502â12513, 2025. 1, 2, 6, 7

[63] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multi-view video diffusion model for 4d generation. arXiv preprint arXiv:2405.20674, 2024. 1, 2

[64] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in Neural Information Processing Systems, 36, 2024. 2

[65] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10324â10335, 2024. 2