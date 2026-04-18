<!-- page 1 -->
LVD-GS: GAUSSIAN SPLATTING SLAM FOR DYNAMIC SCENES VIA HIERARCHICAL
EXPLICIT-IMPLICIT REPRESENTATION COLLABORATION RENDERING
Wenkai Zhu, Xu Li∗, Qimin Xu, Benwu Wang, Kun Wei, Yiming Peng and Zihang Wang
School of Instrument Science and Engineering, Southeast University, Nanjing, China
ABSTRACT
3D Gaussian Splatting SLAM has emerged as a widely used tech-
nique for high-fidelity mapping in spatial intelligence. However, ex-
isting methods often rely on a single representation scheme, which
limits their performance in large-scale dynamic outdoor scenes and
leads to cumulative pose errors and scale ambiguity. To address
these challenges, we propose LVD-GS, a novel LiDAR-Visual
3D Gaussian Splatting SLAM system.
Motivated by the human
chain-of-thought process for information seeking, we introduce a
hierarchical collaborative representation module that facilitates mu-
tual reinforcement for mapping optimization, effectively mitigating
scale drift and enhancing reconstruction robustness. Furthermore, to
effectively eliminate the influence of dynamic objects, we propose a
joint dynamic modeling module that generates fine-grained dynamic
masks by fusing open-world segmentation with implicit residual
constraints, guided by uncertainty estimates from DINO-Depth fea-
tures. Extensive evaluations on KITTI, nuScenes, and self-collected
datasets demonstrate that our approach achieves state-of-the-art
performance compared to existing methods.
Index Terms— 3D Gaussian Splatting, SLAM, Vision Founda-
tion Model, Visual
1. INTRODUCTION
The recent advent of 3D Gaussian Splatting (3DGS) [1, 2, 3]
has enabled high-fidelity photo realistic mapping for autonomous
robotic SLAM systems, which is a core technology for embodied
intelligence[4, 5]. Within this domain, 3D scene representation has
emerged as a critical research frontier, driving the development of
diverse sparse [6, 7, 8] and dense [9, 10, 11] representation method-
ologies that significantly enhance scene understanding.
However, existing 3DGS-SLAM systems exhibit limited per-
formance in complex outdoor scenarios due to reliance on single-
representation constraints [2, 3], facing significant challenges in
large-scale dynamic scenes. The inherent highly dynamics of out-
door scenes, leading to cumulative errors and trajectory drift [12],
which critically degrades Gaussian point cloud initialization essen-
tial for 3DGS performance. Building on prior works [1, 3, 13, 14]
for outdoor 3DGS SLAM, we identify two core challenges: the lim-
itations of single-representation constraints and dynamic object
interference .
On one hand, outdoor scenes provide abundant perceptual cues
derived from highly discriminative features in both semantic and ap-
pearance domains. However, some existing indoor/outdoor 3DGS
∗is the corresponding author.
Email:
lixu.mail@163.com.
This
work was supported in part by the National Key Research and Develop-
ment Program of China under Grant 2022YFB3904404,in part by the Na-
tional Natural Science Foundation of China under Grant 62473099.Website:
https://zwk0901.github.io/LVD-GS2025.
Fig. 1. An overview of the chain-of-thought process, we leverage the
high-level semantic understanding to construct hierarchical explicit-
implicit collaborative representation constraints.
SLAM systems rely primarily on pixel-level photometric or geomet-
ric reconstruction for optimization [1, 2, 15, 16, 17]. This inherent
characteristic leads to lack of higher-level semantic representation
and global feature understanding in unbounded outdoor scenes.
On the other hand, due to the highly dynamic nature of outdoor
environments, the lack of dynamic modeling degrades subsequent
pose estimation and map reconstruction. Although existing methods
attempt to remove these dynamic elements through masking [18, 19,
20], they often apply rigid removal strategies without considering the
loss of feature consistency during ego-motion and lack fine-grained
analysis of dynamic regions. Therefore, these issues raising the fun-
damental question: how to simulate the human chain-of-thought
process to selectively focus on outdoor rich scene information
through explicit-implicit representation.
To address these challenges, we propose LVD-GS SLAM, a
novel LiDAR-Visual Gaussian Splatting SLAM framework designed
for dynamic outdoor scenes. As illustrated in Fig. 1, building on Vi-
sion Foundation Models (VFMs), we propose an advanced represen-
tation collaboration mechanism that facilitates mutual reinforcement
to optimize the mapping process, which effectively resolving scale
ambiguity and enhancing reconstruction fidelity. Subsequently, we
propose a joint dynamic modeling module utilizing open-world seg-
mentation with implicit residual constraints to generate finer-grained
dynamic object masks. The key innovations and contributions of this
paper are highlighted as follows:
(1) We propose a novel LiDAR-Visual 3D Gaussian Splatting
SLAM framework for dynamic scenes, termed LVD-GS, which in-
corporates hierarchical representations collaboration of geometric,
semantic, and DINO feature to effectively higher-level understand-
ing and achieve high-fidelity reconstruction.
(2) We propose a joint dynamic modeling approach that lever-
ages uncertainty estimation from DINO-Depth features, which com-
arXiv:2510.22669v1  [cs.CV]  26 Oct 2025

<!-- page 2 -->
Fig. 2. SGD-GS SLAM System Overview. A large-scale 3D Gaussian Splatting framework incorporating a multi-scale representation
collaboration module, joint dynamic modeling module. We optimize camera poses using L loss to establish initial pose priors, and refine
these poses by incorporating 3D geometric information through scan-to-map registration follows the KISS-ICP[6]. To alleviate memory
constraints, the map is partitioned into localized submaps maintained within a fixed spatial range.
bining open-world segmentation with implicit residual constraints to
produce fine-grained dynamic object masks.
(3)Extensive evaluations on KITTI, nuScenes, and self-collected
datasets demonstrate that our method achieves state-of-the-art per-
formance in both pose estimation accuracy and novel view synthesis
among existing 3DGS-SLAM systems.
2. METHOD
In this section, we will introduce the LVD-GS SLAM pipeline, illus-
trated in Fig. 2. We process RGB frames and LiDAR point clouds
using known camera intrinsics K ∈R3×3. Our framework inte-
grates two core novel modules: (1) Hierarchical Representation Col-
laboration Rendering(Sec. 2.1) (2) Explicit-Implicit Joint Dynamic
Modeling (Sec. 2.2)
2.1. Hierarchical Representation Collaboration Mapping
2.1.1. Hierarchical Representation Extraction
we leverage Grounded SAM [21] -equipped with scene-aware
prompt generation—to extract semantic [22] and DINO features.The
depth features are generated through LiDAR point cloud projection
onto image planes and densified using DepthLab [23]. This integra-
tion builds hierarchical Sem-Geo-DINO representations that unify
semantic, geometric and appearance attributes across multi-scale
spaces, establishing robust consistency constraints.
2.1.2. Representation Collaboration Rendering
To enhance the geometric and photometric fidelity of the Gaussian
map, we propose a Hierarchical Representation Collaboration
Rendering Module optimized using a novel loss function that en-
forces multi-scale consistency between differentiable renderings and
ground truth.
We construct color and depth loss [18] by comparing the ren-
dered RGB and depth values with the ground truth values.
Lc =
1
|M|
|M|
X
i=0
Ci −Cgt
i
 ,
Ldepth =
1
|M|
|M|
X
i=0
Di −Dgt
i

(1)
where Ci, Di are rendered RGB and depth values, Cgt
i , Dgt
i
are
ground truth values.
For supervising semantic information, we employ cross-entropy
loss. Notably, during semantic rendering, we detach the gradient to
prevent this loss from interfering with the optimization of geometry
and appearance features.
Ls = −
X
m∈M
L
X
l=1
pl (m) · log bpl (m)
(2)
where pl represents multi-class semantic probability at class l of the
ground truth map.
To integrate higher-level scene understanding encoded in the
features, we introduce a DINO-feature loss: Ldino, to guide the op-
timization of the enriched scene representation. This loss measures
the feature similarity between the DINO features Fi and the rendered
feature maps F ′
i:
Ldino =
1
Nd
Nd
X
i=0

1 −
Fi · F ′
i
∥Fi∥2 · ∥F ′
i∥2

(3)
where Nd denotes the feature dimension of DINO, and i indexes
the feature vectors. Finally, the complete multi-scale feature loss
function L is the weighted sum of the above losses:
L = λsLs + λdinoLdino + λcLc + λdepthLdepth
(4)

<!-- page 3 -->
Table 1. Pose estimation performance comparison on KITTI and
self-collected datasets. ATE-RMSE is used as the primary metric.
Methods
K03
K05
K06
K07
K09
K10 SC01 SC02
MonoGS[24] 57.27 51.47 93.81 51.23 81.23 61.96 68.43 56.24
SplaTAM[15] 10.31 37.13 53.78 32.82 70.23 33.96 45.12 38.74
OpenGS[16] 19.42 17.39 26.47 14.74 29.31 11.53 20.87 19.73
S3POGS[1]
6.36
5.94
9.34
5.63
8.64
6.52
8.63
7.12
Ours
1.74
1.37
0.69
0.62
2.19
1.45
1.73
1.27
where λs, λdino, λc, λdepth are weighting coefficients.
2.2. Explicit-Implicit Joint Dynamic Modeling
2.2.1. Uncertainty Prediction
we adapt this approach to outdoor dynamic scenes by modeling per-
pixel Gaussian distributions. This uncertainty representation, de-
rived from fused DINO-Depth features, facilitates joint implicit con-
straints across geometric and appearance domains. The residuals U
are defined as:
U = λ′
dinoLdino + λ′
depthLdepth
(5)
We leverage the rapid rendering capability of 3D Gaussian
Splatting (3DGS) to incorporate the residuals U into an objective
function for estimating a per-pixel uncertainty map.
This map
is subsequently thresholded to generate a binary motion mask
Mimplicit(u), which is used to filter dynamic keypoints from
keyframes and prevent their incorporation into the map.
Mimplicit = I
 
min
σ
1
HW
H
X
i=1
W
X
j=1
ρ(Uij, σ)
!
(6)
2.2.2. Refinement of Dynamic masks
To enhance the accuracy and completeness of dynamic object seg-
mentation, we introduce an uncertainty-aware joint modeling ap-
proach that integrates explicit open-world segmentation with im-
plicit residual constraints. This fusion yields more precise dynamic
object masks, formulated as:
Mrefine = Mexplicit ∩Mimplicit
(7)
where Mexplicit denotes the mask obtained from open-world segmen-
tation and Mimplicit represents the mask derived from implicit residual
constraints.
3. EXPERIMENTS
3.1. Implementation and Experiment Setup
We conduct experiments on the nuScenes[25], KITTI[26] and Self-
collected Dataset. To evaluate the rendering performance, we use
PSNR and SSIM metrics to assess the rendered images. And we use
ATE-RMSE(m) to evaluate the pose estimation performance. We
compare our method with SLAM approaches five 3DGS SLAM sys-
tems MonoGS[24], SplaTAM[15], LoopSplat[17], OPENGS[16],
S3POGS[1]. Our implementation is based on the PyTorch frame-
work and tested in NVIDIA RTX3090Ti GPU.
Fig. 3. Trajectory Visualization. Due to the memory constraints,
other 3DGS-SLAM methods can not run to completion on all se-
quences, we present only our method’s trajectory and error.
3.2. Experiment Results
3.2.1. Pose Estimation Results
We evaluate the pose estimation performance of our method on the
KITTI [26] dataset and SC dataset containing urban and campus
scenes with dynamic objects. As summarized in Tab. 1, our ap-
proach demonstrates superior tracking accuracy across all datasets.
By incorporating multi-scale representations and initializing Gaus-
sians from LiDAR points, our system optimizes pose estimation
through multi-level features, providing additional constraints that
enhance model convergence.
Due to memory constraints, other
3DGS-SLAM methods were evaluated only on the first 350 frames
per sequence. However, their tracking threads showed large pose es-
timation errors in outdoor environments, limiting their applicability
in real-world large-scale scenes. S3PO-GS[1] performs relatively
well due to its introduction of pointmap constraints, which effec-
tively mitigate scale drift.
Furthermore, our Hierarchical Representation Collaboration
method enhances the camera pose estimation by capturing accurate,
rich contextual information, thereby achieving more robust local-
ization. As shown in Fig. 3 presents the trajectories of our method
on both the KITTI and self-collected datasets, demonstrating its
consistent performance across different environments. These results
substantiate the overall superiority of the proposed approach.
3.2.2. Novel View Synthesis
As shown in Tab. 2, our method achieves state-of-the-art novel view
synthesis performance across both datasets. Compared to current
3DGS-based SLAM baselines, PSNR shows significant improve-
ments: +4.48 dB on nuScenes , +1.51 dB on KITTI and +3.79 dB
on SC(self-collected). Fig. 4 demonstrates rendered images across
three scenarios(urban, highway and compus). For outdoor environ-
ments, our approach generates photorealistic reconstructions with
enhanced fidelity in vehicle contours, architectural structures, and
road surface details. Notably, in highly dynamic regions, our method

<!-- page 4 -->
Fig. 4. Novel view synthesis results on KITTI (top) , nuScenes(mid) and Self-Collected datasets (bottom). Our approach effectively
handles complex dynamic environments through a Dynamic Modeling module and Representation Collaboration constraints.
Fig. 5. Ablation study. Comparison with two novel modules: Dy-
namic Modeling and Representation Collaboration.
successfully filters transient objects while maintaining scene con-
sistency, which reduces tracking drift and ensures temporal coher-
ence in synthesized sequences. These results demonstrate the capa-
bility of our hierarchical representation collaboration in mitigating
scale drift in outdoor scenes and validate the efficacy of the explicit-
implicit joint dynamic modeling module in complex urban settings.
3.3. Ablation Study
In this section, we evaluate the effectiveness of individual modules
within our proposed LVD-GS framework. As summarized in Ta-
ble 3 and illustrated in Fig. 5, the Dynamic Modeling and Represen-
tation Collaboration components effectively reduce cumulative drift
in outdoor environments. We further compare novel view synthesis
performance between these two novel modules. Our results show
that the Representation Collaboration optimization yields superior
performance in large-scale outdoor scenes, where Sem-Geo-DINO
cues significantly enhance mapping quality.
Table 2. Novel View Synthesis Results on KITTI, nuScenes and
self-collected datasets.Note: P denotes PSNR. S denotes SSIM.
Method
KITTI[26]
nuScenes[25]
SC
P↑
S↑
P↑
S↑
P↑
S↑
MonoGS[24]
14.30
0.441
18.58
0.709
15.76
0.627
SplaTAM[15]
14.62
0.473
18.29
0.723
16.17
0.669
LoopSplat[17]
16.43
0.74
23.07
0.761
18.42
0.754
OPENGS[16]
15.61
0.495
22.04
0.758
17.84
0.741
S3POGS[1]
19.73
0.646
24.25
0.827
21.64
0.780
Ours
21.24
0.81
28.73
0.893
25.43
0.847
Table 3. Ablation Study on Two Core Modules
Dynamic
Representation
PSNR
SSIM
LPIPS
ATE
Modeling
Collaboration
(dB)↑
↑
↓
(m)↓
✗
✗
20.07
0.724
0.577
10.54
✓
✗
22.79
0.780
0.513
8.42
✗
✓
23.27
0.804
0.498
2.97
✓
✓
25.43
0.847
0.340
1.27
4. CONCLUSION
We propose LVD-GS SLAM, a novel LiDAR-visual 3D Gaussian
Splatting system that tackles dynamic scenes and scale drift in out-
door environments. Unlike other 3DGS-based SLAM methods, our
approach uses representations collaboration to constrain mapping
optimization and integrates a joint explicit-implicit module for dy-
namic object removal. Future work we will futher build instance-
level cognitive navigation 3DGS maps.

<!-- page 5 -->
5. REFERENCES
[1] Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and Hao
Wang, “Outdoor monocular slam with global scale-consistent
3d gaussian pointmaps,”
arXiv preprint arXiv:2507.03737,
2025.
[2] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li,
“Gs-slam: Dense visual slam with
3d gaussian splatting,”
in 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024, pp.
19595–19604.
[3] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Os-
wald, “Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting,” arXiv preprint arXiv:2312.10070, 2023.
[4] Lei Ren, Jiabao Dong, Shuai Liu, Lin Zhang, and Lihui Wang,
“Embodied intelligence toward future smart manufacturing in
the era of ai foundation model,” IEEE/ASME Transactions on
Mechatronics, 2024.
[5] Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang, Guan-
bin Li, Wen Gao, and Liang Lin, “Aligning cyber space with
physical world: A comprehensive survey on embodied ai,”
IEEE/ASME Transactions on Mechatronics, 2025.
[6] Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Louis
Wiesmann, Jens Behley, and Cyrill Stachniss, “Kiss-icp: In
defense of point-to-point icp – simple, accurate, and robust reg-
istration if done the right way,” IEEE Robotics and Automation
Letters, vol. 8, no. 2, pp. 1029–1036, 2023.
[7] Han Wang, Chen Wang, Chun-Lin Chen, and Lihua Xie, “F-
loam : Fast lidar odometry and mapping,” in 2021 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS), 2021, pp. 4390–4396.
[8] Chunran Zheng, Wei Xu, Zuhao Zou, Tong Hua, Chongjian
Yuan, Dongjiao He, Bingyang Zhou, Zheng Liu, Jiarong Lin,
Fangcheng Zhu, Yunfan Ren, Rong Wang, Fanle Meng, and
Fu Zhang, “Fast-livo2: Fast, direct lidar–inertial–visual odom-
etry,” IEEE Transactions on Robotics, vol. 41, pp. 326–346,
2025.
[9] Yue Pan, Xingguang Zhong, Louis Wiesmann, Thorbj¨orn
Posewsky, Jens Behley, and Cyrill Stachniss, “Pin-slam: Li-
dar slam using a point-based implicit neural representation for
achieving global map consistency,”
IEEE Transactions on
Robotics, vol. 40, pp. 4045–4064, 2024.
[10] Lin Chen, Boni Hu, Jvboxi Wang, Shuhui Bu, Guangming
Wang, Pengcheng Han, and Jian Chen, “G²-mapping: Gen-
eral gaussian mapping for monocular, rgb-d, and lidar-inertial-
visual systems,”
IEEE Transactions on Automation Science
and Engineering, vol. 22, pp. 12347–12357, 2025.
[11] Sheng Hong, Chunran Zheng, Yishu Shen, Changze Li,
Fu Zhang, Tong Qin, and Shaojie Shen, “Gs-livo: Real-time li-
dar, inertial, and visual multisensor fused odometry with gaus-
sian mapping,” IEEE Transactions on Robotics, vol. 41, pp.
4253–4268, 2025.
[12] Dong Kong, Xu Li, Qimin Xu, Yue Hu, and Peizhou Ni,
“Sc lpr: Semantically consistent lidar place recognition based
on chained cascade network in long-term dynamic environ-
ments,” IEEE Transactions on Image Processing, vol. 33, pp.
2145–2157, 2024.
[13] Renxiang Xiao, Wei Liu, Yushuai Chen, and Liang Hu, “Liv-
gs: Lidar-vision integration for 3d gaussian splatting slam in
outdoor environments,” IEEE Robotics and Automation Let-
ters, vol. 10, no. 1, pp. 421–428, 2025.
[14] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison,
“Gaussian splatting slam,”
in 2024
IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2024, pp. 18039–18048.
[15] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, and Scherer, “Splatam: Splat, track & map
3d gaussians for dense rgb-d slam,” in 2024 IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR),
2024, pp. 21357–21366.
[16] Dianyi Yang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang, and
Mengyin Fu, “Opengs-slam: Open-set dense semantic slam
with 3d gaussian splatting for object-level scene understand-
ing,” arXiv preprint arXiv:2503.01646, 2025.
[17] Liyuan Zhu, Yue Li, Erik Sandstr¨om, Shengyu Huang, Konrad
Schindler, and Iro Armeni, “Loopsplat: Loop closure by reg-
istering 3d gaussian splats,” in 2025 International Conference
on 3D Vision (3DV). IEEE, 2025, pp. 156–167.
[18] Chen Zou, Qingsen Ma, Jia Wang, Ming Lu, Shanghang
Zhang, and Zhaofeng He, “Gaussianenhancer: A general ren-
dering enhancer for gaussian splatting,” in ICASSP 2025 - 2025
IEEE International Conference on Acoustics, Speech and Sig-
nal Processing (ICASSP), 2025, pp. 1–5.
[19] Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng Feng,
and Li Zhang,
“Dg-slam: Robust dynamic gaussian splat-
ting slam with hybrid pose optimization,” Advances in Neu-
ral Information Processing Systems, vol. 37, pp. 51577–51596,
2024.
[20] Hongxing Zhou, Juan Chen, and Zhiqing Li, “Dynamic slam
with 3d gaussian splatting supporting monocular sensing,”
IEEE Sensors Journal, 2025.
[21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang,
Hang Su, et al.,
“Grounding dino:
Marrying dino with
grounded pre-training for open-set object detection,” in Euro-
pean conference on computer vision. Springer, 2024, pp. 38–
55.
[22] Nan Wang, Xiaohan Yan, Xiaowei Song, and Zhicheng Wang,
“Semantic-guided gaussian splatting with deferred rendering,”
in ICASSP 2025 - 2025 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), 2025, pp.
1–5.
[23] Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang,
Hao Ouyang, Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen,
and Ping Luo, “Depthlab: From partial to complete,” arXiv
preprint arXiv:2412.18153, 2024.
[24] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison,
“Gaussian splatting slam,”
in 2024
IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2024, pp. 18039–18048.
[25] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora,
Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi-
ancarlo Baldan, and Oscar Beijbom,
“nuscenes: A multi-
modal dataset for autonomous driving,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recog-
nition, 2020, pp. 11621–11631.
[26] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Ur-
tasun, “Vision meets robotics: The kitti dataset,” The interna-
tional journal of robotics research, vol. 32, no. 11, pp. 1231–
1237, 2013.
