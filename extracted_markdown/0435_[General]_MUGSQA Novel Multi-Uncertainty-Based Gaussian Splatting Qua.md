# MUGSQA: NOVEL MULTI-UNCERTAINTY-BASED GAUSSIAN SPLATTING QUALITY ASSESSMENT METHOD, DATASET, AND BENCHMARKS

Tianang Chen, Jian Jin\*, Shilv Cai, Zhuangzi Li, Weisi Lin\*

Nanyang Technological University

## ABSTRACT

Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and code are available at https://github.com/ Solivition/MUGSQA.

Index Termsâ 3D Gaussian Splatting, Quality Assessment, Dataset, Benchmark

## 1. INTRODUCTION

3D reconstruction is a fundamental problem in computer vision, aiming to recover accurate geometry and appearance of real-world objects and scenes. Among emerging approaches, the first method based on Gaussian Splatting (GS) [1] offers a compelling balance between high rendering quality and real-time performance. Its outstanding performance quickly makes it one of the most promising solutions for practical deployment in 3D object reconstruction and draws attention from both academia and industry.

Although numerous GS-based reconstruction methods [1â6] have recently been proposed, two fundamental questions remain underexplored: i) How well can GS-based reconstruction methods sustain their performance under different input uncertainties [7] (e.g., different numbers of input views, different initial point clouds, and so on)? ii) Are existing quality assessment metrics [8â10] adequate for evaluating such methods? These questions are pivotal not only for enabling fair comparisons among competing methods but also for driving the continuous improvement of reconstruction performance. To answer the above questions, the benchmarks for GS Quality Assessment (GSQA) are required.

Existing quality assessment benchmarks have primarily focused on images [11, 12], point clouds [13, 14], and meshes [15, 16]. Only a few studies [17â20], have constructed GSQA datasets, but these works mainly target compression-induced degradations [17, 20], rather than the more common distortions arising from input uncertainties during GS reconstruction. Such uncertainties include failed occlusion recovery under sparse view density, detail loss due to low-resolution inputs, perspective distortion from changes in view-to-object distance, and structural deviations caused by inaccuracies in the initial point cloud. Consequently, current GSQA datasets are insufficient not only for comprehensive benchmarking of GS-based reconstruction methods, but also for validating the effectiveness of existing quality metrics in capturing distortions induced by these uncertainties. This limitation has further led to stagnation in the development of the GSQA metric design. To address this gap, we systematically introduce Multiple Uncertainties during the data preparation process, adopt various GS-based reconstruction methods, and construct a new Quality Assessment dataset, termed MUGSQA. Unlike prior work that relies on real-world 2D captures, we select OBJ-format mesh models as reconstruction sources [15, 21]. By focusing on single-object scenes, our dataset eliminates interference from multiple coexisting objects, making it more suitable for controlled distortion analysis and metric design.

Besides, existing Subjective Quality Assessment (SQA) methods often present the 3D object to the subjects with a fixed view or a single-distance display [18, 19], making it difficult to reflect the behavior of subjects when dynamically observing Gaussian objects [22] in interactive or immersive scenarios. In order to better align with the above, we propose a unified multi-distance SQA method that guides observers to examine Gaussian objects from various distances and multiple views. Based on this, we conduct a large-scale subjective experiment to collect quality scores for the MUGSQA dataset. We gather 2,452 participants and ultimately obtain over 226,800 valid scores, ensuring that the scores we finally collect are sufficient and reliable. Finally, we construct two benchmarks based on the MUGSQA dataset to evaluate the robustness of GS-based reconstruction methods and the performance of existing objective quality assessment metrics on Gaussian objects. This fills the gap in the current evaluation system in this field and promotes the standardized development of GSQA.

The overview of MUGSQA is shown in Figure 1. In summary, our main contributions include the following points:

â¢ We propose a unified multi-distance SQA method for Gaussian objects to capture the real subjectsâ quality experience.

â¢ We construct the MUGSQA, which is a large-scale Gaussian object dataset taking into account different uncertainties and various GS-based reconstruction methods.

â¢ We construct a benchmark on MUGSQA to evaluate the reconstruction robustness of representative GS-based methods under diverse uncertainties.

â¢ We construct a benchmark on MUGSQA to evaluate the performance of existing quality assessment metrics for GSQA.

<!-- image-->  
Fig. 1. MUGSQA. In Step 1, we select 55 source models, render, and sample on them. During this process, we simulate a total of 54 combinations of uncertainties that might cause differences. In Step 2, we first employ 6 GS-based methods to reconstruct these models. Then, we render all samples and their source models into videos and filter them according to their quality. In Step 3, we utilize these videos and our SQA method to collect quality scores during subjective experiments. In Step 4, we filter the scores and complete the dataset. Finally, we construct two benchmarks aimed at evaluating existing metrics and comparing the robustness of different GS-based reconstruction methods.

## 2. DATA PREPARATION

Source Models. We select 55 mesh models as ground truth from Sketchfab1, which have been demonstrated to have high geometric complexity and high texture quality [15].

Main Set. To obtain the input data required for reconstruction, we first render multi-view images in Blender2, and export them together with camera poses and point clouds in the NeRF Synthetic Dataset [23] format. In accordance with [21], we do not use 3DGS [1] but instead use LightGaussian [2] for reconstruction, ensuring that overall reconstruction quality does not suffer. To generate stimuli of different qualities, we simulate multiple uncertainties that may be encountered during actual data preparation. In our setup, the size of all objects is normalized, so our parameters must meaningfully reflect how the reconstruction degrades on this scale. Inspired by [7], we use the following settings in Blender: (1) View resolution settings: We choose 1080 Ã 1080, 720 Ã 720, and 480 Ã 480 to model different observations. (2) View quantity settings: We use 72, 36, and 9 views. Here, 72 views ensure dense sampling with minimal occlusion, 36 aligns with standard multi-view datasets [24], and 9 simulates realistic sparse-view conditions. The specific positions of the three quantities of views are shown in Figure 2. (3) View-toobject distance settings: 5 m, 2 m, and 1 m correspond respectively to far-range overview, mid-range balanced capture, and close-up focus. (4) Point cloud initialization settings: Randomly sample 105 point clouds from either the model surface or the full scene, which allows us to simulate ideal initialization versus noisy inputs. These values are carefully chosen to match the unit-scale object space and cover a wide range of common distortion factors. They define a wellcontrolled and representative space for evaluating reconstruction algorithms under varied and realistic degradation conditions. Furthermore, to ensure that the quality distribution of the dataset falls within a common range of distortion, we perform data filtering to exclude samples that completely fail to reconstruct. Finally, our MUGSQA dataset contains 1,970 main set samples.

Additional Set. Next, we construct an additional set using more reconstruction methods. For this set, we only use 3 out of the 55 source models, but employ 5 GS-based methods for reconstruction: 3DGS [1], Mip-Splatting [3], Scaffold-GS [4], EAGLES [5], and Octree-GS [6]. This choice is made to keep the subjective experiment manageable while still covering diverse geometric and textural characteristics. All other settings remain consistent with the main set. Similarly, we filter this set and obtain 444 additional set samples. In total, we obtain 1, 970+444 = 2, 414 reconstructed models. Figure 2 shows the overall data generation pipeline.

## 3. SUBJECTIVE QUALITY ASSESSMENT

Method. To fully assess the quality of Gaussian objects, we propose a unified multi-distance SQA method. As shown in Figure 2, we use Blender to render each source model and output a reference video. Then, we process all stimuli using the same views, generating images from these views using the rendering algorithm of the corresponding method and outputting it as a video. Specifically, we choose 3 view-to-object distances $d _ { 0 } = 1 . 2 m , d _ { 1 } = 1 . 5 m$ , $d _ { 2 } = 1$ .8m to render, and define the view-to-object distance $d ( \theta )$ as a function of the view rotation angle $\theta \in [ 0 ^ { \circ }$ , 1080â¦]:

$$
\begin{array} { r } { d ( \theta ) \ = \ d _ { 0 } + ( d _ { 1 } - d _ { 0 } ) \ \mathrm { t r i } \biggl ( \frac { \theta } { 3 6 0 ^ { \circ } } \biggl ) + \bigl ( d _ { 2 } - d _ { 1 } \bigr ) \ \mathrm { t r i } \biggl ( \frac { \theta - 1 8 0 ^ { \circ } } { 7 2 0 ^ { \circ } } \biggl ) , } \end{array}\tag{1}
$$

where $\operatorname { t r i } ( x ) = 1 - { \big | } 1 - 2 \left( x - \lfloor x \rfloor \right)$ . Each video is 30 FPS and has 180 frames. In addition, each video has a uniform resolution of $1 0 8 0 \times 1 0 8 0$ . Note that since the input images used for reconstruction have no background, we manually add a gray background with RGB values of (153, 153, 153) to each frame of the video.

Experiment. To obtain reliable and controllable results [25], we start a crowdsourced project using MTurk 3 and create a scoring interface. As shown in Figure 3, the interface includes three modules: reference video, distortion video, and scoring area. After each pair of videos is played, workers are allowed to slide the scoring bar. In the training stage, a suggested score and a reason corresponding to the distortion will be displayed. After training, participants can enter the test stage of the experiment, during which the suggested scores will no longer be displayed, and the rest of the content remains the same as in the training stage. At the end of the experiment, the scoring results will be automatically uploaded, and after our review, the participants will be paid. Ultimately, 226,800 quality scores are collected and a total of 2,452 participants complete the experiment.

<!-- image-->  
Fig. 2. Data Generation Pipeline. From left to right, the first part represents the process of generating distorted samples and SQA videos; the second and third parts represent the reconstruction input uncertainty rendering settings and the rendering settings for SQA videos in Blender, respectively. The âShareâ in the figure indicates the use of the same camera parameters. The âReconstruction\*â and âSplatting\*â steps in the figure represent the use of the corresponding algorithm based on the selected GS-based reconstruction method.

<!-- image-->  
Fig. 3. Scoring Interface.

## 4. DATA PROCESSING AND ANALYSIS

## 4.1. Dataset Completion and Comparison

To extract a sufficient and accurate set of valid scores, we adopt the following three-step filtering process. (1) Filter by training stage scores: If a participantâs ranking of the scores of the three samples in the training stage does not match the order of the suggested scores, all scores of the current participant in this playlist will be filtered out. (2) Filter by score distribution: We refer to the ITU-R BT.500-13 screening procedure [26] to detect unreasonable score distributions. This procedure is the same as summarized in [27]. (3) Filter by GUs: Based on the Golden Units (GUs) in each playlist, we perform further score filtering. Unlike the approach in [28], which filters after mapping to discrete values, we retain the original scores and filter according to the distribution of each score list.

<table><tr><td>Name</td><td>Year</td><td>Distortion Factor</td><td>SQA Views</td><td>Ns</td><td> $N _ { o }$ </td><td> $N _ { g }$ </td><td>Nm</td></tr><tr><td>GSC-QA [17]</td><td>2024</td><td>Compression</td><td>360Â°</td><td>9</td><td>6</td><td>120</td><td>1</td></tr><tr><td>NVSQA [18]</td><td>2025</td><td>I</td><td>360+Front</td><td>13</td><td>I</td><td>65</td><td>3</td></tr><tr><td>GS-QA [19]</td><td>2025</td><td>I</td><td>360Â°+Front</td><td>8</td><td>1</td><td>64</td><td>7</td></tr><tr><td>3DGS-IEval-15K [20]</td><td>2025</td><td>Compression</td><td>20 Views</td><td>10</td><td>I</td><td>760</td><td>6</td></tr><tr><td>MUGSQA (Ours)</td><td>2025</td><td>Input Settings</td><td>1, 080Â°</td><td>I</td><td>55</td><td>2,414</td><td>6</td></tr></table>

Table 1. Dataset Comparison. $N _ { s } , N _ { o } , N _ { g } , N _ { m }$ refer to the number of source scenes, source objects, labeled gaussians, and GS-based reconstruction methods, respectively.

As a result, we retain 101,555 valid scores, ensuring that each sample in every playlist has at least 30 valid scores. Then we compute Mean Opinion Scores (MOS) by averaging the scores given by different participants on each stimulus. Similarly to [17], we map the MOS to a continuous range of 0 to 5, where higher scores represent better quality. At this point, we have completed the dataset.

As shown in Table 1, our dataset has several advantages over existing datasets. Firstly, MUGSQA compensates for the deficiencies in GT by using synthetic data. This not only includes the image data required for reconstruction, but also contains the 3D mesh models, providing more reliable comparisons and analyses. Secondly, MUGSQA addresses the shortcomings of existing datasets in singleobject reconstruction. Most datasets only contain scenes, whereas our dataset encompasses 55 synthetic objects as source models. In fact, if a single object can be reconstructed, it will be more conducive to an in-depth analysis of the distortion characteristics and metric design. This need to assess the quality of a single Gaussian object is crucial in scenarios requiring a large number of high-quality synthetic objects [29]. In terms of SQA methods, compared to other datasets that only render frames in a fixed scale, our dataset takes into account the quality differences generated by rendering at different scales, thereby achieving 180 rendered frames and covering as many as 3 cycles. In terms of data annotation, our subjective experiments are also more thorough, including 2,414 valid MOS.

<table><tr><td>Method</td><td> $R _ { o v e r a l l }$ </td><td> $R _ { r e s o l u t i o n }$ </td><td> $R _ { q u a n t i t y }$ </td><td> $R _ { d i s t a n c e }$ </td><td> $R _ { p c }$ </td></tr><tr><td>3DGS [1]</td><td>71.04</td><td>73.04</td><td>69.32</td><td>66.17</td><td>75.62</td></tr><tr><td>Mip-Splatting [3]</td><td>73.06</td><td>73.77</td><td>71.70</td><td>68.35</td><td>78.42</td></tr><tr><td>LightGaussian [2]</td><td>71.42</td><td>72.95</td><td>70.83</td><td>67.08</td><td>74.82</td></tr><tr><td>EAGLES [5]</td><td>70.41</td><td>70.70</td><td>70.88</td><td>64.87</td><td>75.20</td></tr><tr><td>Octree-GS [6]</td><td>66.30</td><td>66.06</td><td>65.78</td><td>65.66</td><td>67.70</td></tr><tr><td>Scaffold-GS [4]</td><td>63.41</td><td>60.56</td><td>71.67</td><td>53.45</td><td>67.95</td></tr></table>

Table 2. Robustness Comparison on MUGSQA Dataset with Per-Column Best (Bold) and Second-best (Underlined) Values.

## 4.2. Robustness of GS-based Reconstruction Methods

To evaluate the robustness of different GS-based reconstruction methods using the MUGSQA dataset, we define a robustness score $R _ { u } \in [ 0 , 1 0 0 ]$ , which integrates three aspects: stability, consistency, and performance [30]. Stability is derived from the coefficient of variation $\begin{array} { r } { C V \ = \ \frac { \sigma } { \nu } \times 1 0 0 \% } \end{array}$ , consistency from the MOS range $M = \operatorname* { m a x } _ { i } \{ M O S _ { i } \} - \operatorname* { m i n } _ { i } \{ M O S _ { i } \}$ , and performance from the mean MOS $\mu .$ These are mapped to [0, 100] and combined as:

$$
\begin{array} { l } { R _ { u } = 0 . 4 \times \operatorname* { m a x } ( 0 , 1 0 0 - 2 \times C V ) + } \\ { 0 . 3 \times \operatorname* { m a x } ( 0 , 1 0 0 - 2 0 \times M ) + 0 . 3 \times \operatorname* { m i n } ( 1 0 0 , 1 0 \times \mu ) ^ { . } } \end{array}\tag{2}
$$

This score is computed for each u independently while keeping the others fixed, where u is the uncertainty settings introduced in Section $^ { 2 , }$ so u â {resolution, quantity, distance, pc}. The final robustness $R _ { o v e r a l l }$ is obtained by averaging $R _ { u }$ across different settings. As shown in Table 2, Mip-Splatting achieves the highest $R _ { o v e r a l l } .$ while 3DGS, EAGLES and LightGaussian also show strong performance. However, Octree-GS and Scaffold-GS, designed for largescene reconstruction, perform poorly in object reconstruction. We believe that optimizations in multi-scale rendering, as well as the coarse-to-fine training strategy, are key to improving the quality of Gaussian object reconstruction and the robustness of the algorithm. Conversely, some methods, such as Level-of-Detail (LOD), while having a more powerful upper limit in reconstruction range, will have their steps correspondingly affected when facing non-ideal input conditions, thereby leading to more severe distortion.

## 4.3. Performance of Objective Quality Assessment Metrics

Our dataset possesses rich 2D and 3D visual data, where quality can be assessed for different modalities. However, unlike data in point cloud or mesh formats [13â16], quality assessment metrics specifically designed for the 3D modality of GS are still lacking. Therefore, we only use 2D metrics for benchmarking.

Metrics. We select several representative Full-Reference (FR) and No-Reference (NR) Image Quality Assessment (IQA) metrics. Specifically, we select 12 FR metrics: PSNR, PSNR-Y, SSIM, SSIM-C, MS-SSIM, CW-SSIM, FSIM, GMSD, NLPD, VSI, LPIPS (VGG), LPIPS (AlexNet) [8], and 4 NR metrics: NIQE, PIQE, DBCNN [9], FID. All these metrics are calculated using IQA-PyTorch4. It is worth noting that because some metrics are based on deep learning, their values are results computed after performing a five-fold cross-validation on the target dataset.

Results and Evaluation. For the results of each metric, if they are not within the specified MOS range, we use a four-parameter logistic regression to map them. Next, we calculate the correlation coefficients between each metric and MOS, including Pearson Linear Correlation Coefficient (PLCC), Spearman Rank-order Correlation Coefficient (SROCC), Root Mean Square Error (RMSE), and

<table><tr><td colspan="5"></td><td colspan="4">Additional Set</td></tr><tr><td>Metric</td><td>PLCC</td><td></td><td>SROCC RMSE</td><td>KROCC</td><td>PLCC</td><td>SROCCI</td><td>RMSE</td><td>KROCC</td></tr><tr><td>PSNR</td><td>0.5848</td><td>0.5246</td><td>1.1026</td><td>0.3662</td><td>0.5146</td><td>0.4749</td><td>1.2873</td><td>0.3266</td></tr><tr><td>PSNR-Y</td><td>0.5865</td><td>0.5260</td><td>1.1009</td><td>0.3674</td><td>0.5264</td><td>0.4937</td><td>1.2765</td><td>0.3379</td></tr><tr><td>SSIM</td><td>0.3686</td><td>0.3695</td><td>1.2635</td><td>0.2496</td><td>0.3148</td><td>0.2724</td><td>1.4251</td><td>0.2023</td></tr><tr><td>SSIM-C</td><td>0.3660</td><td>0.3609</td><td>1.2650</td><td>0.2438</td><td>0.3161</td><td>0.2580</td><td>1.4244</td><td>0.1949</td></tr><tr><td>FSIM</td><td>0.6769</td><td>0.6776</td><td>1.0005</td><td>0.4787</td><td>0.5662</td><td>0.5774</td><td>1.2375</td><td>0.4030</td></tr><tr><td>MS-SSIM</td><td>0.6240</td><td>0.6354</td><td>1.0622</td><td>0.4448</td><td>0.5691</td><td>0.5769</td><td>1.2346</td><td>0.3995</td></tr><tr><td>CW-SSIM</td><td>0.7186</td><td>0.7350</td><td>0.9453</td><td>0.5304</td><td>0.7089</td><td>0.7235</td><td>1.0590</td><td>0.5306</td></tr><tr><td>GMSD</td><td>0.5176</td><td>0.5479</td><td>1.1630</td><td>0.3799</td><td>0.5955</td><td>0.5861</td><td>1.2062</td><td>0.3983</td></tr><tr><td>NLPD</td><td>0.5936</td><td>0.5947</td><td>1.0939</td><td>0.4137</td><td>0.5202</td><td>0.4990</td><td>1.2823</td><td>0.3454</td></tr><tr><td>VSI</td><td>0.7209</td><td>0.7262</td><td>0.9421</td><td>0.5252</td><td>0.6150</td><td>0.6248</td><td>1.1840</td><td>0.4371</td></tr><tr><td>LPIPS-V</td><td>0.4051</td><td>0.4090</td><td>1.2428</td><td>0.2769</td><td>0.5017</td><td>0.5081</td><td>1.2988</td><td>0.3690</td></tr><tr><td>LPIPS-A</td><td>0.4165</td><td>0.4428</td><td>1.2358</td><td>0.3009</td><td>0.5276</td><td>0.5455</td><td>1.2755</td><td>0.3893</td></tr><tr><td>NIQE</td><td>0.1656</td><td>0.0444</td><td>1.3405</td><td>0.0348</td><td>0.1777</td><td>0.1540</td><td>1.4775</td><td>0.1049</td></tr><tr><td>PIQE</td><td>0.0991</td><td>0.0126</td><td>1.3526</td><td>0.0088</td><td>0.2166</td><td>0.1838</td><td>1.4658</td><td>0.1261</td></tr><tr><td>DBCNN</td><td>0.8846</td><td>0.8800</td><td>0.6693</td><td>0.6927</td><td>0.9223</td><td>0.9075</td><td>0.6183</td><td>0.7301</td></tr><tr><td>FID</td><td>0.4782</td><td>0.5156</td><td>1.1938</td><td>0.3578</td><td>0.7585</td><td>0.7680</td><td>0.9784</td><td>0.5722</td></tr></table>

Table 3. Performance Comparison on MUGSQA Dataset with Per-Column Best (Bold) and Second-best (Underlined) Values. LPIPS-V refers to LPIPS (VGG), and LPIPS-A refers to LPIPS (AlexNet).

Kendall Rank Correlation Coefficient (KROCC). Table 3 shows the overall performance of each metric under the two subsets. Among the FR-IQA metrics, except for CW-SSIM and VSI, which perform relatively well, the rest of the metrics yield poor results, and even the LPIPS series, capable of extracting deep features, has difficulty distinguishing the quality of our dataset samples. There are many influencing factors, such as the presence of pure color or empty backgrounds affecting the calculation results of some metrics, the difficulty in distinguishing quality differences after sample filtering, and the features extracted by some metrics from pre-trained DNNs not aligning with the characteristics of GS distortion. These factors collectively lead to the deterioration of the correlation coefficient results of these IQA metrics. For NR-IQA metrics, traditional NIQE and PIQE metrics perform very poorly, clearly indicating that their calculation methods are not suitable for assessing the quality of Gaussian objects. For the more advanced metric, DBCNN, it is able to achieve good results after fine-tuning. This demonstrates the importance of deep learning in modern quality assessment and the powerful ability of these architectures for fine-grained distinction. Based on these results, we find that the IQA metrics that only use 2D rendering results are not sufficient to evaluate the quality of Gaussian objects. Therefore, we call for the design of new metrics specifically for the GS modality, and we believe that if novel metrics can be further optimized to handle the Gaussian attribute designs of different methods, the effectiveness and speed of GSQA will reach even higher levels.

## 5. CONCLUSION

In this paper, we propose a unified multi-distance SQA method. Based on this, we construct a large-scale Gaussian object reconstruction dataset, MUGSQA, and establish two brand new benchmarks through SQA experiment, post-analysis, and filtering. By evaluating the performance of various GS-based reconstruction methods and various existng metrics on this benchmark, we believe that designing new GSQA metrics and conducting a deeper distortion analysis from a multi-modal perspective is an urgent need.

## 6. ACKNOWLEDGEMENT

This research is partially supported by the Ministry of Education, Singapore, under the funding of MOE-T2EP20123-0006.

## 7. REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â TOG, vol. 42, pp. 1â14, 2023.

[2] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al., âLightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS,â in Proc. NeurIPS, 2024, pp. 140138â140158.

[3] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger, âMip-Splatting: Alias-free 3D Gaussian Splatting,â in Proc. CVPR, 2024, pp. 19447â19456.

[4] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai, âScaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering,â in Proc. CVPR, 2024, pp. 20654â20664.

[5] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava, âEA-GLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS,â in Proc. ECCV, 2024, pp. 54â71.

[6] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai, âOctree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians,â TPAMI, pp. 1â15, 2025.

[7] Marcus Klasson, Riccardo Mereu, Juho Kannala, and Arno Solin, âSources of Uncertainty in 3D Scene Reconstruction,â in Proc. ECCVW, 2024, pp. 271â289.

[8] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, âThe Unreasonable Effectiveness of Deep Features as a Perceptual Metric,â in Proc. CVPR, 2018, pp. 586â595.

[9] Weixia Zhang, Kede Ma, Jia Yan, Dexiang Deng, and Zhou Wang, âBlind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network,â TCSVT, vol. 30, pp. 36â47, 2020.

[10] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtai Zhai, and Weisi Lin, âQ-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels,â in Proc. ICML, 2024, pp. 54015â54029.

[11] Vlad Hosu, Hanhe Lin, Tamas Sziranyi, and Dietmar Saupe, âKonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment,â TIP, vol. 29, pp. 4041â4056, 2020.

[12] Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, and Alan Bovik, âFrom Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality,â in Proc. CVPR, 2020, pp. 3575â3585.

[13] Qi Yang, Hao Chen, Zhan Ma, Yiling Xu, Rongjun Tang, and Jun Sun, âPredicting the Perceptual Quality of Point Cloud: A 3D-to-2D Projection-Based Exploration,â TMM, vol. 23, pp. 3877â3891, 2020.

[14] Yipeng Liu, Qi Yang, Yiling Xu, and Le Yang, âPoint Cloud Quality Assessment: Dataset Construction and Learning-based No-reference Metric,â TOMM, vol. 19, pp. 1â26, 2023.

[15] Yana Nehme, Johanna Delanoy, Florent Dupont, Jean-Philippe Â´ Farrugia, Patrick Le Callet, and Guillaume Lavoue, âTex- Â´ tured Mesh Quality Assessment: Large-scale Dataset and Deep Learning-based Quality Metric,â TOG, vol. 42, pp. 1â20, 2023.

[16] Bingyang Cui, Qi Yang, Kaifa Yang, Yiling Xu, Xiaozhong Xu, and Shan Liu, âSJTU-TMQA: A Quality Assessment Database for Static Mesh with Texture Map,â in Proc. ICASSP, 2024, pp. 7875â7879.

[17] Qi Yang, Kaifa Yang, Yuke Xing, Yiling Xu, and Zhu Li, âA Benchmark for Gaussian Splatting Compression and Quality Assessment Study,â in Proc. MMAsia, 2024, pp. 1â8.

[18] Yuhang Zhang, Joshua Maraval, Zhengyu Zhang, Nicolas Ramin, Shishun Tian, and Lu Zhang, âEvaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes,â 2025.

[19] Pedro Martin, Antonio Rodrigues, Jo Â´ ao Ascenso, and Ë Maria Paula Queluz, âGS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis,â 2025.

[20] Yuke Xing, Jiarui Wang, Peizhi Niu, Wenjie Huang, Guangtao Zhai, and Yiling Xu, â3DGS-IEval-15K: A Large-scale Image Quality Evaluation Database for 3D Gaussian-Splatting,â 2025.

[21] Qi Ma, Yue Li, Bin Ren, Nicu Sebe, Ender Konukoglu, Theo Gevers, Luc Van Gool, and Danda Pani Paudel, âA Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining,â in Proc. 3DV, 2025.

[22] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian, âGaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting,â TOG, vol. 43, pp. 1â13, 2024.

[23] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng, âNeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,â CACM, vol. 65, pp. 99â106, 2021.

[24] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir Mech, and Ulrich Neumann, âDISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction,â in Proc. NeurIPS, 2019, pp. 492â502.

[25] Jian Jin, Xingxing Zhang, Xin Fu, Huan Zhang, Weisi Lin, Jian Lou, and Yao Zhao, âJust Noticeable Difference for Deep Machine Vision,â TCSVT, vol. 32, no. 6, pp. 3452â3461, 2021.

[26] B Series, âMethodology for the subjective assessment of the quality of television pictures,â Recommendation ITU-R BT, vol. 500, 2012.

[27] RafaÅ K Mantiuk, Anna Tomaszewska, and RadosÅaw Mantiuk, âComparison of Four Subjective Methods for Image Quality Assessment,â in Proc. CGF, 2012, pp. 2478â2491.

[28] Tobias Hossfeld, Christian Keimel, Matthias Hirth, Bruno Gardlo, Julian Habigt, Klaus Diepold, and Phuoc Tran-Gia, âBest Practices for QoE Crowdtesting: QoE Assessment With Crowdsourcing,â TMM, vol. 16, pp. 541â558, 2013.

[29] William Ljungbergh, Bernardo Taveira, Wenzhao Zheng, Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christoffer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan, âR3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation,â 2025.

[30] Nu Sun, Jian Jin, Lili Meng, Weisi Lin, Hao Wang, Li Liu, and Huaxiang Zhang, âMFCQA: Multi-Range Feature Cross-Attention Mechanism for No-Reference Image Quality Assessment,â KBS, vol. 310, pp. 113027, 2025.