# 3DGS-VBench: A Comprehensive Video Quality Evaluation Benchmark for 3DGS Compression

Yuke Xing1, William Gordon2, Qi Yang3, Kaifa Yang1, Jiarui Wang1, Yiling Xu1

1{xingyuke-v, sekiroyyy, wangjiarui, yl.xu}@sjtu.edu.cn, 2williamg.research@gmail.com, 3qiyang@umkc.edu

1 Shanghai Jiao Tong University, Shanghai, China

2 Basis Independent Silicon Valley, San Jose, CA, USA

3 University of Missouri-Kansas City, Kansas City, USA

Abstractâ3D Gaussian Splatting (3DGS) enables real-time novel view synthesis with high visual fidelity, but its substantial storage requirements hinder practical deployment, prompting state-of-the-art (SOTA) 3DGS methods to incorporate compression modules. However, these 3DGS generative compression techniques introduce unique distortions lacking systematic quality assessment research. To this end, we establish 3DGS-VBench, a large-scale Video Quality Assessment (VQA) Dataset and Benchmark with 660 compressed 3DGS models and video sequences generated from 11 scenes across 6 SOTA 3DGS compression algorithms with systematically designed parameter levels. With annotations from 50 participants, we obtained MOS scores with outlier removal and validated dataset reliability. We benchmark 6 3DGS compression algorithms on storage efficiency and visual quality, and evaluate 15 quality assessment metrics across multiple paradigms. Our work enables specialized VQA model training for 3DGS, serving as a catalyst for compression and quality assessment research. The dataset is available at https://github.com/YukeXing/3DGS-VBench.

Index Termsâ3D-Gaussian-Splatting (3DGS), Video Quality Assessment Dataset and Benchmark, 3DGS Compression

## I. INTRODUCTION

3D Gaussian Splatting (3DGS) [1] has emerged as a transformative approach for novel view synthesis (NVS), offering real-time rendering capabilities and visual fidelity compared to NeRF [2]. However, its explicit representation demands substantial storage resources, severely hindering practical deployment, which prompting state-of-the-art (SOTA) 3DGS algorithms [3]â[11] to incorporate compression modules to balance visual quality and storage efficiency. These generative compression techniques introduce unique distortions that necessitate specialized quality assessment (QA) models. Thus, the scale and diversity of quality evaluation datasets for training and evaluation play a crucial role in developing accurate and robust 3DGS compression and QA models.

As we know, there have been 7 studies on NVS QA dataset, experiencing a transition from NeRF-oriented to 3DGS-centered evaluation. NeRF-QA [12] and NeRF-VSQA [13], featuring 48 and 88 video samples respectively, while FFV [14] established pairwise evaluation protocols using 220 samples. Systematic distortion analysis was advanced by ENeRF-QA [15], which incorporated NeRF compression methodologies across 440 samples and characterized 9 distinct NeRF-related distortion categories. The emergence of 3DGS as a dominant NVS paradigm prompted corresponding shifts in evaluation priorities. Compression-related QA was explored by GSC-QA [16] using 120 samples to analyze their specific

3DGS compression approach, while GS-QA [17] conducted comparative analysis between 3DGS and NeRF methodologies across 64 samples. Both video and image evaluation domains were addressed by NVS-QA [18], incorporating 65 samples for each modality.

However, existing 3DGS QA datasets exhibit some critical limitations hindering comprehensive evaluation. (1) Limited distortion types. Despite compression becoming essential for 3DGS practical deployment due to substantial storage demands, most current datasets predominantly fail to incorporate diverse generative compression strategies. This oversight results in insufficient samples with diverse compressioninduced distortion effects for effective metric training. (2) Limited dataset scale. Due to substantial costs of subjective experiments, existing datasets contain fewer than 100 samples with the largest not exceeding 500. More critically, none of the existing datasets comprise over 100 authentic real-world scene samples, severely insufficient for training robust deep learningbased QA models that simulate human perception standards.

To address these limitations, we present 3DGS-VBench, a large-scale dataset and benchmark for 3DGS video quality evaluation, containing 660 3DGS models and corresponding video sequences. The contents consist of 11 real-world scenes selected from mainstream multi-view datasets [19]â[21]. 6 representative mainstream 3DGS compression algorithms [3]â [8] with systematically designed multi-level compression parameters are considered, leading to diverse distortion and large sample scale. After rendering the 3DGS samples into videos and conducting subjective experiments, we collect 10k expert ratings to calculate mean opinion scores (MOS). Based on 3DGS-VBench, we benchmark the 6 representative 3DGS models across storage efficiency and visual quality dimensions. Besides, leveraging the substantial scale of our dataset, we conduct a comprehensive QA metrics evaluation benchmark with diverse IQA/VQA metrics across multiple evaluation paradigms on 3DGS content. This benchmark reveals the limitations of existing QA algorithms, as well as the characteristics of SOTA 3DGS compression models in terms of storage efficiency and visual quality. The main contributions of this paper are summarized as follows:

â¢ We establish 3DGS-VBench, a large-scale dataset comprising 660 3DGS-rendered videos with diverse 3DGS compression distortions annotated with MOS scores.

â¢ Based on 3DGS-VBench, we benchmark and highlight the characteristics of SOTA 3DGS compression methods from two performance dimensions: storage efficiency and visual quality.

<!-- image-->  
Fig. 1: 11 selected scenes in 3DGS-VBench: scenes 1-5 depict outdoor scenes, while scenes 6-11 depict indoor scenes.

â¢ We evaluate and benchmark 15 QA metrics on 3DGS-VBench. Thorough analysis is provided, we report the weakness of current 3DGS QA study.

## II. DATASET CONSTRUCTION

## A. Source Content Selection

To achieve comprehensive quality assessment with diverse visual characteristics, we select 11 real-world scenes from 4 well-established multiview datasets as shown in Fig. 1. Six scenes are picked from Mip-NeRF360 [19], including three outdoor scenes: bicycle (1237 Ã 822), flowers (1256 Ã 828), and garden (1297 Ã 840), and three indoor scenes: counter (1558 Ã 1038), kitchen (1558 Ã 1039), and room (1557 Ã 1038). Two outdoor scenes are selected from Tanks & Temples [20]: train (980 Ã 545) and truck (979 Ã 546). Two indoor scenes are selected from Deep Blending [21]: playroom (1264 Ã 832) and drjohnson (1332 Ã 876). One human figure scene Dance Dunhuang Pair (dance) with resolution 1600 Ã 876 is selected from PKU-DyMVHumans [22].

## B. Camera Parameter Setup for Video Rendering

Existing classical 3DGS real scene multi-view image datasets [19]â[21] consist of discrete camera viewpoints, divided into training and testing sets. The training set optimizes 3DGS models, while the testing set provides reference images for quality assessment of rendered views from the reconstructed scene. However, video rendering requires continuous camera trajectories, which are absent in these multiview datasets containing only sparse, discrete viewpoints. This necessitates generating dense, continuous camera paths for video evaluation, which we name it as âValâ viewpoint set.

To create the âValâ set, first, we train 3DGS models to obtain pointcloud representations of selected scenes. Then, we import these pointclouds into Blender where we establish virtual camera setups with carefully calibrated parameters including focal length, position, orientation, and resolution settings. Through iterative adjustment, we design 600 continuous viewpoints for each scene, maintaining resolution consistency with original datasets. These viewpoints form a 360â¦ orbital trajectory for smooth video sequence generation.

## C. 3DGS Model Selection & Compression Parameter Design

To systematically investigate visual distortion effects in 3DGS compression, we select 6 representative mainstream algorithms and analyze their compression strategies and key parameters. For each compression parameter, we establish multiple Compression Levels (CL) to control distortion intensity. For each 3DGS training process, we vary only one compression parameter with different CL while other compression parameters maintain the default optimal values. This methodology can generate 3DGS models with diverse distortion levels (DL) and types, establishing comprehensively study for visual distortion effects introduced by different 3DGS compression strategies. The selected 3DGS algorithms and detailed CL configurations are presented as follows:

a) Compact-3DGS [8]: achieves compression through learnable volume masking (lambda parameter), residual vector quantization (RVQ) for geometry encoding (codebook size and RVQ depth), and hash grid-based neural fields replacing spherical harmonics (SH) (hashmap parameter). As illustrated in table I, with 3 key compression parameters each having 5 CL where CL5 represents the default optimal configuration, we have 3 compression parameters Ã 4 distorted CL = 12 distorted samples, and 1 default sample with all optimal parameters, resulting in 13 DL per scene.

TABLE I: CL for compression parameters in Compact-3DGS.
<table><tr><td>Parameter</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5 (default)</td></tr><tr><td>hashmap</td><td>21</td><td>22</td><td>27</td><td>212</td><td>219</td></tr><tr><td>(codebook, rvq_depth)</td><td>(23, 1)</td><td>(24, 1)</td><td>(26, 1</td><td>(23, 6)</td><td>(26, 6)</td></tr><tr><td>lambda</td><td>0.014</td><td>0.012</td><td>0.010</td><td>0.006</td><td>0.0005</td></tr></table>

b) CompGS [6]: employs K-means vector quantization of Gaussian parameters, controlled by geometry codebook size (g-size), color codebook size (c-size), and opacity regularization parameter (reg). As shown in table II, with 3 key compression parameters and 5 CL each, we generate 3 Ã 4 + 1 = 13 DL per scene.

TABLE II: CL for compression parameters in CompGS.
<table><tr><td>Parameter</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5 (default)</td></tr><tr><td>g-size</td><td>1</td><td>2</td><td>23</td><td>25</td><td>212</td></tr><tr><td>c-size</td><td>1</td><td>2</td><td>22</td><td>23</td><td>212</td></tr><tr><td>reg</td><td>4Ã 1e-6</td><td>3Ã 1e-6</td><td>2Ã 1e-6</td><td>1 Ã 1e-6</td><td>1 Ã 1e-7</td></tr></table>

c) c3dgs [7]: uses K-means to construct color and geometry codebooks with entropy encoding. Compression is controlled by codebook size and importance thresholds (csize/c-include for color, g-size/g-include for geometry). As illustrated in table III, with 2 key compression parameters each having 5 CL, we generate $\mathbf { 2 } \times \mathbf { 4 } + \mathbf { 1 } = \mathbf { 9 }$ DL per scene.

TABLE III: CL for compression parameters in c3dgs.
<table><tr><td>Parameter</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5 (default)</td></tr><tr><td>(c-size, c-include)</td><td>(1, 0.6)</td><td>(2, 0.6)</td><td>(22, 0.6)</td><td>(25, 0.6)</td><td>(212, 0.6Ã 1e-6)</td></tr><tr><td>(g-size, g-include)</td><td>(1, 0.3)</td><td>(23,0.3)</td><td>(25 0.3)</td><td>(28, 0.3)</td><td>(212, 0.3Ã 1e-5)</td></tr></table>

d) LightGaussian [5]: compresses through pruning lowsignificance Gaussians, distilling SH coefficients to lower degrees and quantizing them via vector quantization. Compression is controlled by pruning ratio (prune) and SH quantization parameters (c-ratio/c-size). As shown in table IV, we generate $\mathbf { 2 } \times 4 + \mathbf { 1 } = \mathbf { 9 }$ DL per scene.

TABLE IV: CL for parameters in LightGaussian.
<table><tr><td>Parameter</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5 (default)</td></tr><tr><td>prune (c-ratio, c-size)</td><td>0.97 (1, 2)</td><td>0.95 (1, 22)</td><td>0.90 (1, 23)</td><td>0.85 (1, 25)</td><td>0.66 (1, 213)</td></tr></table>

e) Scaffold-GS [3]: uses anchor-based hierarchical sampling, with compression controlled by voxel size parameter (vsize). As shown in Table V, with 1 compression parameter corresponding to 8 CL, we generate 8 DL per scene.

TABLE V: CL for compression parameter in Scaffold-GS.
<table><tr><td>Param</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5</td><td>CL6</td><td>CL7</td><td>CL8 (default)</td></tr><tr><td>vsize</td><td>0.350</td><td>0.250</td><td>0.200</td><td>0.160</td><td>0.120</td><td>0.080</td><td>0.050</td><td>0.001</td></tr></table>

f) HAC [4]: besides using anchor-based hierarchical sampling, they employs context-aware compression with adaptive quantization controlled by the lambda parameter to balance rate-distortion performance. As shown in Table VI, with 1 compression parameter corresponding to 8 CL, we generate 8 DL per scene.

TABLE VI: CL for compression parameter in HAC.
<table><tr><td>Param</td><td>CL1</td><td>CL2</td><td>CL3</td><td>CL4</td><td>CL5</td><td>CL6</td><td>CL7</td><td>CL8 (default)</td></tr><tr><td>lmbda</td><td>0.800</td><td>0.600</td><td>0.400</td><td>0.300</td><td>0.200</td><td>0.120</td><td>0.060</td><td>0.004</td></tr></table>

In all, we trained 11 scenes $\times ~ ( 1 3 + 1 3 + 9 + 9 + 8 +$ 8) model settings = 660 3DGS reconstruction models.

## D. Subjective Experiment and Data Processing

For subjective QA, we convert each 3DGS reconstruction into Processed Video Sequences (PVS) by rendering 600 frames that orbit each scene with uniform angular spacing established in our âValâ camera trajectories. The frames are encoded into 20-second PVS using FFMPEG with libx265 codec at 30 fps and a constant rate factor of 10.

For the quality annotation, we use an 11-level impairment scale proposed by ITU-TP.910 [23]. The experiment was carried out using a 27-inch AOC Q2790PQ monitor in an indoor laboratory environment under standard lighting conditions. The videos are displayed using an interface designed with Python Tkinter. To prevent visual fatigue caused by excessively experiment time, 660 PVSs are randomly divided into 8 smaller groups. Finally, we obtain a total of 9,900 human annotations (15 annotators Ã 660 videos).

We follow the suggestions recommended by ITU to conduct the outlier detection and subject rejection. The score rejection rate is 2%. At last, a total of 660 MOSs are obtained.

## III. DATASET VALIDATION

## A. Diversity of Source Content

To assess the diversity of selected source content, we analyzed the spatial information (SI) [23] and temporal information (TI) [23] characteristics of our 11 scenes, as presented in Fig. 2 (a), where the labels match the scene numbers in Fig. 1. The scattered distribution pattern demonstrates comprehensive coverage across both spatial and temporal complexity dimensions, indicating effective diversity in scene characteristics. Thus, our dataset provides a robust foundation for evaluating 3DGS across diverse visual content types.

## B. Analysis of MOS

To verify the reasonability of MOS scores, Fig. 2 (b) shows the MOS distribution covers the full 0-10 range with an approximately Gaussian distribution centered at 6. The dataset maintains adequate representation across all quality segments, including severely degraded content (0-2), providing essential training diversity for robust QA model development.

<!-- image-->

<!-- image-->  
(b)  
Fig. 2: (a) SI vs. TI (b) MOS Distribution

## C. Analysis of CL Design

To validate our MOS scores and compression level settings, we examine the MOS-storage relationship using LightGaussian as example and evaluating key compression parameters across five designed CL in Table IV on four representative scenes: truck (Tanks & Temples), drjohnson (Deep Blending), flowers and room (Mip-NeRF360).

<!-- image-->

<!-- image-->  
Fig. 3: MOS vs. Storage (MB) for LightGaussian with diverse CL settings. (a) 5 CL for parameter prune. (b) 5 CL for parameters (c-ratio, c-size).

TABLE VII: Comprehensive Quantitative Comparison Benchmark for 6 representative 3DGS compression methods in 3 Realworld Multi-view images Datasets. The best results are marked in RED and the second-best in BLUE .
<table><tr><td>Dataset</td><td colspan="5">Mip-NeRF360 [19]</td><td colspan="5">Tanks&amp;Temples [20]</td><td colspan="5">Deep Blending [21]</td></tr><tr><td>Methods / Metrics</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPSâ</td><td>MOSâ</td><td>Size(MB)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>MOSâ</td><td>Size(MB)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>MOSâ</td><td>Size(MB)â</td></tr><tr><td>3DGS (origin) [1]</td><td>27.6655</td><td>0.9141</td><td>0.1267</td><td>-</td><td>795.1323</td><td>23.7579</td><td>0.89339</td><td>0.09559</td><td>-</td><td>434.3680</td><td>29.6219</td><td>0.9269</td><td>0.1016</td><td>-</td><td>665.9170</td></tr><tr><td>Compact-3DGS [8]</td><td>27.0677</td><td>0.9006</td><td>0.1494</td><td>8.1186</td><td>48.7238</td><td>23.3552</td><td>0.8803</td><td>0.1139</td><td>6.9000</td><td>39.7017</td><td>29.7882</td><td>0.9280</td><td>0.1016</td><td>8.6786</td><td>43.3126</td></tr><tr><td>c3dgs [7]</td><td>27.3182</td><td>0.9067</td><td>0.1374</td><td>8.2960</td><td>28.6562</td><td>23.6088</td><td>0.8880</td><td>0.1003</td><td>7.5619</td><td>17.6700</td><td>29.5096</td><td>0.9250</td><td>0.0968</td><td>8.6786</td><td>23.8754</td></tr><tr><td>LightGaussian [5]</td><td>25.4406</td><td>0.8582</td><td>0.2009</td><td>8.1333</td><td>52.0151</td><td>22.6449</td><td>0.8565</td><td>0.1637</td><td>7.4190</td><td>28.6682</td><td>25.9883</td><td>0.8447</td><td>0.1950</td><td>7.9333</td><td>43.3730</td></tr><tr><td>CompGS [6]</td><td>27.1742</td><td>0.9042</td><td>0.1468</td><td>8.3484</td><td>21.9188</td><td>23.2633</td><td>0.8845</td><td>0.1132</td><td>7.5524</td><td>13.6345</td><td>30.0751</td><td>0.9338</td><td>0.0991</td><td>9.0238</td><td>14.7798</td></tr><tr><td>Scaffold-GS [3]</td><td>27.8406</td><td>0.9149</td><td>0.1285</td><td>8.8233</td><td>166.8277</td><td>242154</td><td>09033</td><td>0.0868</td><td>8.5595</td><td>147.3997</td><td>30.2592</td><td>0.9349</td><td>0.0933</td><td>8.7308</td><td>111.8945</td></tr><tr><td>HaC [4]</td><td>27.5911</td><td>0.9112</td><td>0.1364</td><td>8.7413</td><td>15.8625</td><td>24.3093</td><td>0.9014</td><td>0.0928</td><td>8.1786</td><td>14.5386</td><td>30.3207</td><td>0.9342</td><td>0.0975</td><td>8.8214</td><td>7.7961</td></tr></table>

The results for geometry parameter prune are shown in Fig. 3 (a), while Fig. 3 (b) shows results for color compression parameters (c-ratio, c-size). The clear monotonic relationships between storage size and MOS scores validate our scoring system and CL design, as larger models should inherently store more information and achieve higher visual quality.

Besides, the results reveal distinct compression behaviors: In Fig. 3 (a), prune achieves significant storage reduction with minimal quality loss (CL3-CL5), while in Fig. 3 (b), (c-ratio, c-size) cause substantial quality degradation with limited storage savings (CL1-CL4). This demonstrates that for LightGaussian, prune provides superior quality-storage tradeoffs for practical deployment.

## IV. BENCHMARK FOR QUALITY ASSESSMENT METRICS

To evaluate existing QA methods on compressed 3DGS content, we conduct correlation analysis between objective metrics and MOS scores using Spearman Rank-order Correlation Coefficient (SRCC), Pearson Linear Correlation Coefficient (PLCC), and Kendall Rank Correlation Coefficient (KRCC).

TABLE VIII: Performance benchmark on 3DGS-VBench. â¡ various IQA models, â  deep-learning-based VQA models.
<table><tr><td>Metrics</td><td>|| Ref</td><td>SRCC</td><td>PLCC</td><td>KRCC</td></tr><tr><td>PSNR (TIP 2004) [24] 3 SSIM (TIP 2004) [24] 3 LPIPS (CVPR 2018) [25] 3 DISTS (PAMI 2022) [26] 3 VIF (TIP 2006) [27] 3 FSIM (TIP 201) [28] 3 IW-SSIM (TIP 2011) [29] MS-SSIM (SSC 2003) [30]</td><td>sss</td><td>0.5022 0.5108 0.5106 0.7317 0.4220 05866 0.6309 0.5028</td><td>0.4976 0.4758 0.4581 0.7146 0.4252 0.5815 0.5962</td><td>0.3560 0.3684 0.3619 0.5269 0.2917 0.4311</td></tr><tr><td>3 CLIP-IQA (AI 2023) [31] 3 BRISQUE (TIP 2012) [32] . DOVER (ICCV 2023) [33] FAST-VQA (ECCV 2022) [34] simpleVQA (ICM 2022) [35] VSFA (ICM 2019) [36] Q-Align (PAMI 2023) [37]</td><td>X X X xxxx</td><td>0.3913 0.2379 0.9409 0.9314 0.9350 0.9392 0.8485</td><td>0.4700 0.2738 0.1819 0.9308 0.9255 0.7813 0.9345</td><td>0.4488 0.3602 0.3216 0.11749 0.7901 0.7753 0.9314 0.7908</td></tr></table>

We test 15 objective metric: i) full-reference image metrics, including classic metrics: PSNR, SSIM [24], MS-SSIM [30], IW-SSIM [29], VIF [27] and FSIM [28], and deeplearning based metrics: LPIPS [25] and DISTS [26]; ii) noreference image metrics, including BRISQUE [32], and LLMpretraining metric CLIP-IQA [31]; iii) deep learning-based no-reference video metrics, including DOVER [33], FAST-VQA [34] simpleVQA, and Q-Align [37], as thereâs often no reference videos in typical multi-view image datasets.

As shown in Table VIII, deep learning-based VQA models substantially outperform traditional approaches, with DOVER achieving highest correlation (SRCC=0.9409, PLCC=0.9308,

KRCC=0.7901). SimpleVQA, VSFA, and FAST-VQA also show strong performance (about 0.93 SRCC). Among traditional metrics, DISTS performs best (SRCC=0.7317), while no-reference methods show poor performance (BRISQUE SRCC=0.2379). Since current 3DGS generation uses L1 loss and SSIM supervision, we think using better metrics as loss functions [38] could improve generation results.

## V. COMPRESSION BENCHMARK FOR 3DGS MODELS

To investigate current 3DGS compression performance across visual quality and storage efficiency, we comprehensively compared 6 representative methods on 3 datasets using default optimal parameters, establishing the first standard benchmark for 3DGS compression algorithms.

As shown in Table VII, anchor-based method Scaffold-GS achieves superior visual quality exceeding original 3DGS but limited compression (4.8Ã to 5.9Ã). HAC provides optimal balance with aggressive compression (50Ã to 85Ã) while maintaining competitive quality (< 0.3 dB loss vs. Scaffold-GS), validating entropy codingâs effectiveness.

Among component-wise methods, CompGS also shows balanced performance with moderate quality loss (< 1 dB) and reasonable compression (15Ã to 45Ã). LightGaussian demonstrates relatively unsatisfactory trade-offs with significant quality degradation (2 â 4 dB PSNR loss) despite moderate compression (12Ã to 15Ã).

Our benchmark shows modern 3DGS compression achieves 10Ã to 85Ãstorage reduction with < 1 dB quality loss for top methods. HAC provides optimal quality-efficiency balance for practical deployment, though additional encoding/decoding time is required, while Scaffold-GS suits qualitycritical applications. Recent Anchor-based approaches outperform component-wise compression methods.

## VI. CONCLUSION

We present 3DGS-VBench, the first comprehensive VQA dataset for compressed 3DGS, comprising 660 models and video sequences from 11 scenes across 6 algorithms with systematically designed compression parameters. Through subjective evaluation, we obtain MOS scores and validate dataset diversity and reliability. Our evaluation of 15 IQA/VQA metrics reveals limitations on 3DGS-specific distortions and establishes a standardized benchmark comparing 6 algorithms across storage efficiency and visual quality. This work enables specialized VQA model training and opens possibilities for robust learnable QA metrics and 3DGS compression optimization strategies.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, p. 99â106, 2021.

[3] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20654â20664, 2024.

[4] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, âHac: Hash-grid assisted context for 3d gaussian splatting compression,â in European Conference on Computer Vision (ECCV), pp. 422â438, Springer, 2024.

[5] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang, et al., âLightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,â Advances in neural information processing systems, vol. 37, pp. 140138â140158, 2024.

[6] K. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and H. Pirsiavash, âCompgs: Smaller and faster gaussian splatting with vector quantization,â in European Conference on Computer Vision, pp. 330â349, Springer, 2024.

[7] S. Niedermayr, J. Stumpfegger, and R. Westermann, âCompressed 3d gaussian splatting for accelerated novel view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10349â10358, June 2024.

[8] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, âCompact 3d gaussian splatting for static and dynamic radiance fields,â arXiv preprint arXiv:2408.03822, 2024.

[9] S. Girish, K. Gupta, and A. Shrivastava, âEagles: Efficient accelerated 3d gaussians with lightweight encodings,â in European Conference on Computer Vision, pp. 54â71, Springer, 2024.

[10] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, âCompgs: Efficient 3d scene representation via compressed gaussian splatting,â in Proceedings of the 32nd ACM International Conference on Multimedia, pp. 2936â2944, 2024.

[11] Q. Yang, L. Yang, G. Van Der Auwera, and Z. Li, âHybridgs: High-efficiency gaussian splatting data compression using dualchannel sparse representation and point cloud encoder,â arXiv preprint arXiv:2505.01938, 2025.

[12] P. Martin, A. Rodrigues, J. Ascenso, and M. P. Queluz, âNerf-qa: Neural radiance fields quality assessment database,â in 2023 15th International Conference on Quality of Multimedia Experience (QoMEX), pp. 107â 110, 2023.

[13] P. Martin, A. Rodrigues, J. Ascenso, and M. Paula Queluz, âNerf view synthesis: Subjective quality assessment and objective metrics evaluation,â IEEE Access, vol. 13, pp. 26â41, 2025.

[14] H. Liang, T. Wu, P. Hanji, F. Banterle, H. Gao, R. Mantiuk, and C. Oztireli, âPerceptual quality assessment of nerf and neural viewÂ¨ synthesis methods for front-facing views,â in Computer Graphics Forum, vol. 43, p. e15036, Wiley Online Library, 2024.

[15] Y. Xing, Q. Yang, K. Yang, Y. Xu, and Z. Li, âExplicit-nerf-qa: A quality assessment database for explicit nerf model compression,â in 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP), pp. 1â5, 2024.

[16] Q. Yang, K. Yang, Y. Xing, Y. Xu, and Z. Li, âA benchmark for gaussian splatting compression and quality assessment study,â in Proceedings of the 6th ACM International Conference on Multimedia in Asia, pp. 1â8, 2024.

[17] P. Martin, A. Rodrigues, J. Ascenso, and M. P. Queluz, âGs-qa: Comprehensive quality assessment benchmark for gaussian splatting view synthesis,â arXiv preprint arXiv:2502.13196, 2025.

[18] Y. Zhang, J. Maraval, Z. Zhang, N. Ramin, S. Tian, and L. Zhang, âEvaluating human perception of novel view synthesis: Subjective quality assessment of gaussian splatting and nerf in dynamic scenes,â arXiv preprint arXiv:2501.08072, 2025.

[19] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 5470â5479, 2022.

[20] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[21] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, âDeep blending for free-viewpoint image-based rendering,â ACM Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1â15, 2018.

[22] X. Zheng, L. Liao, X. Li, J. Jiao, R. Wang, F. Gao, S. Wang, and R. Wang, âPku-dymvhumans: A multi-view video benchmark for highfidelity dynamic human modeling,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22530â 22540, 2024.

[23] P. ITU-T RECOMMENDATION, âSubjective video quality assessment methods for multimedia applications,â 1999.

[24] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, âImage quality assessment: From error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[25] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586â595, 2018.

[26] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, âImage quality assessment: Unifying structure and texture similarity,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 5, pp. 2567â 2581, 2022.

[27] H. Sheikh and A. Bovik, âImage information and visual quality,â IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430â444, 2006.

[28] L. Zhang, L. Zhang, X. Mou, and D. Zhang, âFsim: A feature similarity index for image quality assessment,â IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378â2386, 2011.

[29] Z. Wang and Q. Li, âInformation content weighting for perceptual image quality assessment,â IEEE Transactions on Image Processing, vol. 20, no. 5, pp. 1185â1198, 2011.

[30] Z. Wang, E. Simoncelli, and A. Bovik, âMultiscale structural similarity for image quality assessment,â in The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003, vol. 2, pp. 1398â1402 Vol.2, 2003.

[31] J. Wang, K. C. Chan, and C. C. Loy, âExploring clip for assessing the look and feel of images,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, pp. 2555â2563, 2023.

[32] A. Mittal, A. K. Moorthy, and A. C. Bovik, âNo-reference image quality assessment in the spatial domain,â IEEE Transactions on Image Processing, vol. 21, no. 12, pp. 4695â4708, 2012.

[33] H. Wu, E. Zhang, L. Liao, C. Chen, J. Hou, A. Wang, W. Sun, Q. Yan, and W. Lin, âExploring video quality assessment on user generated contents from aesthetic and technical perspectives,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20144â 20154, 2023.

[34] H. Wu, C. Chen, J. Hou, L. Liao, A. Wang, W. Sun, Q. Yan, and W. Lin, âFast-vqa: Efficient end-to-end video quality assessment with fragment sampling,â in European conference on computer vision, pp. 538â554, Springer, 2022.

[35] W. Sun, X. Min, W. Lu, and G. Zhai, âA deep learning based noreference quality assessment model for ugc videos,â in Proceedings of the 30th ACM International Conference on Multimedia, pp. 856â865, 2022.

[36] D. Li, T. Jiang, and M. Jiang, âQuality assessment of in-the-wild videos,â in Proceedings of the 27th ACM international conference on multimedia, pp. 2351â2359, 2019.

[37] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun, et al., âQ-align: Teaching lmms for visual scoring via discrete text-defined levels,â arXiv preprint arXiv:2312.17090, 2023.

[38] Q. Yang, Y. Zhang, S. Chen, Y. Xu, J. Sun, and Z. Ma, âMped: Quantifying point cloud distortion based on multiscale potential energy discrepancy,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 6037â6054, 2023.