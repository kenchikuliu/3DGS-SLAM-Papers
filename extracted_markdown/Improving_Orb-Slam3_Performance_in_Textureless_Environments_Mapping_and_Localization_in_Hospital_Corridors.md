<!-- page 1 -->
Improving ORB-SLAM3 Performance in
Textureless Environments: Mapping and
Localization in Hospital Corridors
Zijun Sha,∗,˜Member,˜IEEE, Syuuhei Shiro†, Kazuki Shibamiya†, Kazuhiro Shintani∗and Kazuhiro Tanaka∗
∗Frontier Research Center
Toyota Motor Corporation, Toyota, Japan
Email: zijun sha@mail.toyota.co.jp
†KSK CO.,LTD., Tokyo, Japan
Abstract—Simultaneous Localization and Mapping (SLAM)
plays a pivotal role in robotics applications, particularly for
autonomous navigation in indoor environments. However, hos-
pital corridors present unique challenges for traditional SLAM
systems due to their textureless surfaces and homogeneous ap-
pearances. This study proposes an enhanced visual SLAM frame-
work that integrates machine learning-based feature extraction
(DISK) and matching (LightGlue) into ORB-SLAM3 to address
these challenges. Extensive experiments conducted in Toyota
Memorial Hospital demonstrate significant improvements in both
mapping and localization performance. In mapping tasks, our
system achieves 100% success rate compared to ORB-SLAM3’s
80%, while reducing mapping error by 13.9%. For localization
tasks, our system maintains robust performance across different
trajectories, achieving a 90.3% tracking rate compared to ORB-
SLAM3’s 23.5% in challenging scenarios. The localization error
is reduced by 81.1% and 50.9% in same-trajectory and different-
trajectory scenarios, respectively. These results demonstrate that
our enhanced framework successfully addresses the challenges of
visual SLAM in hospital environments, providing a more reliable
solution for autonomous robot navigation in healthcare facilities.
Index Terms—Visual SLAM, Deep Learning, Feature Extrac-
tion, Feature Matching, Mapping, Localization
I. INTRODUCTION
In the coming years, autonomous delivery robots will be
widely deployed in hospitals, shopping malls, and other in-
door environments [1]. Currently, 2D LiDAR-based navigation
systems are widely used in indoor environment due to their
simplicity and robustness, and we have also successfully
implemented such system on Fig. 1(a) for transportation of
medication in Toyota Memorial Hospital [2]. However, 2D
LiDAR systems face inherent limitations, including restricted
environmental perception and challenges in adapting to com-
plex 3D environments. These constraints have motivated the
exploration of visual SLAM approaches, which offer richer
environmental context and adaptability to diverse scenarios.
Meanwhile, Visual SLAM systems, particularly ORB-SLAM3
[3], have shown promising results in various applications.
These has motivated the exploration to deploy vision-based
navigation for hospital environment.
In this study, we focus on Visual SLAM-based mapping
and localization in hospital corridor environments, particularly
(a) Potaro [2].
(b) HSR [4].
(c) Homogeneous scenes in hospital corridor.
(d) Mapping trajectory in long corridor.
Fig. 1: Hospital long corridor mapping
in Toyota Memorial Hospital. Hospital corridor environments
present 3 unique challenges for visual SLAM systems:
1) Unlike typical indoor environments, hospital corridors
present extremely textureless regions, making robust
feature extraction particularly difficult.
2) The homogeneous appearances of walls and floors char-
acteristic in Fig. 1(c) make traditional feature matching
unreliable.
3) High reliability in both mapping and localization is
required to ensure the safety and efficiency of robot
723
2025 IEEE Conference on Artificial Intelligence (CAI)
979-8-3315-2400-5/25/$31.00 ©2025 IEEE
DOI 10.1109/CAI64502.2025.00130
2025 IEEE Conference on Artificial Intelligence (CAI) | 979-8-3315-2400-5/25/$31.00 ©2025 IEEE | DOI: 10.1109/CAI64502.2025.00130
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:34 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
navigation.
To further evaluate ORB-SLAM3’s performance in these chal-
lenging conditions, we conducted experiments using the Hu-
man Support Robot (HSR) [4] (Fig. 1(b)) in Toyota Memorial
Hospital corridors. The tests allowed us to validate ORB-
SLAM3’s ability to navigate and map long corridors under
textureless conditions. The results confirmed the same chal-
lenges observed earlier: mapping distortions and localization
failures occurred frequently, particularly in long corridors with
insufficient keypoints or repetitive patterns. As illustrated in
Fig. 1(d), the ORB-SLAM3 trajectory deviated significantly
from the ground truth obtained using 2D-LiDAR-based MCL
localization on a pre-built map. Although this is an extreme
example, similar deviations were frequently observed through-
out our experiments. This drift mainly results from the lack of
distinctive features in low-texture scenes and the occurrence
of incorrect matches in homogeneous, repetitive environments.
Such issues necessitate improvements in feature extraction and
matching methods to enhance visual SLAM reliability and
accuracy.
In recent years, machine learning–based feature extraction
methods [5], [6] and matching techniques [7], [8] have ad-
vanced significantly. In particular, DISK [6] delivers consid-
erable improvements in sparse-feature environments, while
LightGlue [8] offers high-precision matching with a speed
advantage. Prior study [9] has demonstrated that incorporating
SuperPoint [5] into ORB-SLAM3 enhances mapping and
localization performance. These motivate our integration of
DISK and LightGlue into ORB-SLAM3 to boost its perfor-
mance in hospital environments.
II. METHODOLOGY
Input Data
Stereo 
Camera
IMU
Back-End
Pose
Map
Python Wrapper Interface
Feature Matching
(LightGlue)
Feature Extraction
(DISK)
Tracking
Keyframe
processing
Front-End
Local 
Mapping
Loop Closing
Global 
Optimization
Fig. 2: Framework of our SLAM
Fig. 2 illustrates the framework of our enhanced visual
SLAM system. The system consists of three key steps to
achieve robust performance in textureless environments.
In the first step, we establish a Python wrapper interface for
ORB-SLAM3, enabling seamless integration of state-of-the-art
machine learning-based feature extraction and matching algo-
rithms. This wrapper maintains the core SLAM functionality
while providing flexible API endpoints for feature processing.
In the second step, we replace the traditional ORB feature
extraction with DISK and implement LightGlue for feature
matching. DISK generates more stable and distinctive fea-
tures even in textureless regions, while LightGlue ensures
robust matching across different viewpoints and conditions.
To accommodate these new features, we regenerated the Bag
of Words (BoW) vocabulary using the DISK descriptors,
ensuring optimal performance in loop closure detection and
relocalization.
In the third step, we integrate IMU measurements with the
enhanced visual features. The system fuses visual and inertial
data through a tightly-coupled optimization framework, where
IMU data compensates for the lack of visual cues in featureless
environments. This fusion strategy particularly benefits scenar-
ios where visual features become temporarily unreliable due
to homogeneous surfaces or sudden movements.
The modular design of our system allows for rapid evalua-
tion and integration of different feature extraction and match-
ing algorithms. Through the Python wrapper interface, new
algorithms can be tested without modifying the core SLAM
architecture, enabling continuous improvement as better meth-
ods become available.
III. EXPERIMENTS AND RESULTS
A. Experimental Setup
The framework was evaluated in Toyota Memorial Hospital
corridors using the HSR platform equipped with a stereo
camera (MYNT EYE S with 752×480 resolution at 60 fps),
a IMU (Analog Devices ADIS16475), and a 2D LiDAR
(Hokuyo UTM-30LX-EW). The experiments were performed
on a system powered by an Intel Core i9-14900KF CPU and
an NVIDIA RTX 4090 GPU. All hardware and processing
parameters were kept constant across trials. Due to space
constraints at the hospital, all data were collected on site
and subsequently tested offline. Given ORB-SLAM3’s wide
adoption, our study focuses on a direct comparison with it.
Furthermore, prior work [9] showed that integrating Super-
Point can significantly improve ORB-SLAM3 performance.
Based on this insight, we extended our Python wrapper API
to incorporate SuperPoint for feature extraction while also
employing LightGlue for matching. This setup enables us
to directly compare our system against both the standard
ORB-SLAM3 and its SuperPoint-LightGlueenhanced variant
(SPLG). To evaluate the performance of mapping and local-
ization separately, we conducted two distinct test sets:
1) Mapping: 10 trials following identical trajectory, focus-
ing on the system’s mapping consistency and reliability.
2) Localization: Two test scenarios, each repeated 10 times:
a) Same Trajectory (S.T.): Following identical path as
mapping to evaluate system consistency
b) Different Trajectory (D.T.): Following alternative
path while maintaining same motion direction to
assess system adaptability.
724
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:34 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
We evaluated performance using the Absolute Trajectory
Error (ATE)-RMSE and tracking rate metrics. Additionally,
we measured the processing rate to quantify the computational
overhead introduced by integrating DISK, SuperPoint, and
LightGlue.
B. Mapping Analysis
In terms of consistency and reliability, we define a mapping
failure as a trial in which the average ATE-RMSE exceeds
approximately 1 m. In hospital environments, where high
mapping accuracy is critical for safety, such a threshold is
particularly important. Based on this definition, TABLE I
shows that SPLG and our system achieved a 100% mapping
success rate across 10 trials, while ORB-SLAM3 experienced
2 severe failures with maps exhibiting significant distortion or
incompleteness. This clearly demonstrates the superior robust-
ness of the machine learning–based approach in challenging
hospital environments.
TABLE I : Mapping successful rate
Success
Fail
Successful rate
ORB-SLAM3
8
2
80%
SPLG
10
0
100%
Ours
10
0
100%
The quantitative evaluation using ATE-RMSE metric is
presented in TABLE II. In mapping tasks, our system achieves
an average error of 0.427 m, a 13.9% improvement over ORB-
SLAM3’s 0.496 m and SPLG’s 0.461 m. Furthermore, the
reduced standard deviation (0.119 m vs. 0.173 m for ORB-
SLAM3 and 0.130 m for SPLG) reflects more consistent
performance across multiple trials. The maximum error is
also decreased from 0.805 m (ORB-SLAM3) to 0.641 m,
demonstrating enhanced stability in challenging scenarios.
TABLE II : ATE-RMSE [m] metric on Mapping
Mean±std
Max.
Min.
Improvment
ORB-SLAM3
0.496±0.173
0.805
0.294
-
SPLG
0.461±0.130
0.646
0.197
7.06%
Ours
0.427±0.119
0.641
0.210
13.9%
TABLE III demonstrates that our approach, compared to
ORB-SLAM3 and SPLG, generates fewer map points while
significantly increasing the number of keyframes, resulting in a
lower average number of points per frame. This indicates more
efficient feature utilization and superior keyframe distribution,
which ultimately contributes to improved mapping stability.
TABLE III : Map Points and Keyframes
Map Points
Keyframes
Avg Points/Frame
ORB-SLAM3
45185
418
108
SPLG
24252
667
36
Ours
19576
1036
19
All approaches are maintained high tracking rates during
mapping, as shown in TABLE IV. Our system achieved a
99.85% tracking rate, nearly matching ORB-SLAM3’s per-
fect tracking. This confirms that integrating machine learn-
ing–based approaches does not compromise tracking consis-
tency while enhancing mapping accuracy and stability.
TABLE IV : Tracking rate on Mapping
Tracking rate
ORB-SLAM3
100.00%
SPLG
99.81%
Ours
99.85%
Fig. 3 showcases a visual and quantitative comparison
between ORB-SLAM3 and our system in Toyota Memorial
Hospital’s corridor. As suggested by Fig. 3(a) and TABLE III,
our system—despite using fewer feature points—may deliver
significantly higher mapping accuracy, potentially by capturing
more stable and uniformly distributed features. Fig. 3(b)
further demonstrates that our trajectory closely aligns with the
ground truth, in contrast to the noticeable deviation in ORB-
SLAM3. These results highlight the enhanced stability and
precision of our approach in textureless environments.
ORB-SLAM3
Ours
(a) Feature Tracking comparison.
(b) Mapping trajectory in long corridor.
Fig. 3: Feature Tracking and Mapping trajectory comparison
C. Localization Analysis
The tracking rate analysis, presented in TABLE V, reveals
a stark contrast in system robustness under both Different
Trajectory (D.T.) and Same Trajectory (S.T.) scenarios. In the
D.T. scenario, ORB-SLAM3 and SPLG exhibit low tracking
rates (23.5% and 36.14%, respectively), while our system
achieves a much higher rate of 90.3%. Under the S.T. scenario,
all methods perform well, with tracking rates of 95.63%
(ORB-SLAM3), 97.03% (SPLG), and 98.35% (ours). These
results indicate that our approach offers significantly improved
robustness and adaptability, especially in challenging condi-
tions with varying viewpoints.
TABLE V : Tracking rate on Localization
Tracking rate (D.T.)
Tracking rate (S.T.)
ORB-SLAM3
23.5%
95.63%
SPLG
36.14%
97.03%
Ours
90.3%
98.35%
725
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:34 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
Table VI shows that in the S.T. scenario, ORB-SLAM3 has
an average ATE-RMSE of 2.213 m. SPLG reduces this error
to 0.466 m (a 79.0% improvement), while our method further
lowers it to 0.419 m (an 81.1% improvement). These results
indicate that our approach slightly outperforms SPLG in terms
of localization accuracy under the same trajectory conditions,
demonstrating more consistent and reliable tracking perfor-
mance.
TABLE VI : ATE-RMSE [m] on Localization of S.T.
Mean±std
Max.
Min.
Improvment
ORB-SLAM3
2.213±0.655
2.879
0.914
-
SPLG
0.466±0.133
0.667
0.202
79.0%
Ours
0.419±0.117
0.634
0.2
81.1%
Table VII shows that in the DT scenario, ORB-SLAM3
yields a mean error of 0.850 m (std 0.635 m, max 1.531 m).
SPLG reduces this error to 0.242 m (std 0.128 m), a 71.5%
improvement, while our method achieves 0.417 m (std 0.119
m, max 0.645 m) for a 50.9% improvement. It is noteworthy
that, as shown in Table V, both ORB-SLAM3 and SPLG have
tracking rates below 36.1% in the D.T. scenario, underscoring
the challenge of reliable tracking under these conditions. In
contrast, our method not only enhances localization accuracy
but also significantly improves tracking performance.
TABLE VII : ATE-RMSE [m] on Localization of D.T.
Mean±std
Max.
Min.
Improvement
ORB-SLAM3
0.850±0.635
1.531
0.073
-
SPLG
0.242±0.128
0.550
0.124
71.5%
Ours
0.417±0.119
0.645
0.209
50.9%
D. Processing Rates Analysis
TABLE VIII : Processing Rates
Extraction(Hz)
Matching(Hz)
Overall Cycle(Hz)
ORB-SLAM3
66.7
1000
22.7
SPLG
28.6
62.5
11.2
Ours
8.1
71.4
5.8
Table VIII presents the processing rates for feature ex-
traction, matching, and overall cycle. ORB-SLAM3 achieves
66.7 Hz for extraction and 1000 Hz for matching, resulting
in a 22.7 Hz cycle. In contrast, SPLG records 28.6 Hz for
extraction and 62.5 Hz for matching, corresponding to an
11.2 Hz overall cycle, while our approach registers 8.1 Hz
for extraction and 71.4 Hz for matching, with a 5.8 Hz cycle.
These results highlight the trade-off between improved feature
quality and increased computational cost. While both SPLG
and our method enhance feature robustness compared to ORB-
SLAM3, they also result in slower processing speeds, which
may impact localization and mapping performance in high-
speed motion scenarios. In particular, the use of DISK for
feature extraction significantly slows down the system, making
it the primary bottleneck in processing speed.
IV. CONCLUSION
This study demonstrates that integrating machine learn-
ing–based feature extraction and matching into ORB-SLAM3
significantly improves visual SLAM performance in hospi-
tal corridors. Testing at Toyota Memorial Hospital revealed
notable gains in both mapping and localization compared
to conventional ORB-SLAM3 and the SuperPoint-enhanced
variant (SPLG).
For mapping, our system achieved a 100% success rate
(80% for ORB-SLAM3), reduced mean ATE-RMSE by
13.9%, and lowered error variance. Its higher keyframe count
indicates more efficient feature use and improved stability.
Compared to SPLG, our method achieved even greater error
reduction and stability.
In localization, our system reduced the mean ATE-RMSE
by 81.1% in the same trajectory scenario and achieved a 90.3%
tracking rate in the different trajectory scenario, compared to
23.5% for ORB-SLAM3 (36.14% for SPLG). These results
underscore our system’s superior robustness and adaptability.
Our processing rates analysis further reveals that the en-
hanced feature robustness comes at the cost of increased
computational overhead. In particular, the DISK-based feature
extraction significantly slows down the system, emerging as
the primary processing bottleneck—a factor that may impact
real-time performance in high-speed scenarios.
In future work, we will extend the framework to dynamic
settings and optimize computational efficiency to overcome
current processing limitations, enabling real-time performance
on resource-constrained platforms.
REFERENCES
[1] G. Fragapane, R. De Koster, F. Sgarbossa, and J. O. Strandhagen,
“Planning and control of autonomous mobile robots for intralogistics:
Literature review and research agenda,” Eur. J. Oper. Res., vol. 294, no. 2,
pp. 405–426, 2021.
[2] T. F. R. C. of Toyota Motor Corporation, “Robots carrying medicine in
place of humans!” [Online]. Available: https://global.toyota/en/mobility/
frontier-research/40390293.html, [Accessed: Dec. 17, 2024].
[3] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D. Tard´os,
“Orb-slam3: An accurate open-source library for visual, visual–inertial,
and multimap slam,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890,
2021.
[4] T. Yamamoto, K. Terada, A. Ochiai, F. Saito, Y. Asahara, and K. Murase,
“Development of human support robot as the research platform of a
domestic mobile manipulator,” ROBOMECH J., vol. 6, no. 1, pp. 1–15,
2019.
[5] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Self-
supervised interest point detection and description,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2018, pp.
224–236.
[6] M. Tyszkiewicz, P. Fua, and E. Trulls, “Disk: Learning local features
with policy gradient,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 14 254–
14 265, 2020.
[7] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Super-
glue: Learning feature matching with graph neural networks,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4938–
4947.
[8] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “Lightglue: Local feature
matching at light speed,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.
(ICCV), 2023, pp. 17 627–17 638.
[9] G. Mollica, M. Legittimo, A. Dionigi, G. Costante, and P. Valigi,
“Integrating sparse learning-based feature detectors into simultaneous
localization and mapping—a benchmark study,” Sensors, vol. 23, no. 4,
p. 2286, 2023.
726
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:34 UTC from IEEE Xplore.  Restrictions apply.
