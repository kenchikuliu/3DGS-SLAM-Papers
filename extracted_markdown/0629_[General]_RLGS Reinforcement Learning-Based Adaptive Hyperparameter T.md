<!-- page 1 -->
RLGS: Reinforcement Learning-Based Adaptive Hyperparameter Tuning for
Gaussian Splatting
Zhan Li, Huangying Zhan, Changyang Li, Qingan Yan, Yi Xu
Goertek Alpha Labs
{first.last}@goertekusa.com
Ground Truth
Count (M)
TMGS + Ours
TMGS
Figure 1: Our reinforcement learning hyperparameter tuning boosts the state-of-the-art Taming 3DGS (TMGS) across varying
number of Gaussians. In the Train scene where adding Gaussians fails after 0.7 million, our method still scales up quality.
Abstract
Hyperparameter tuning in 3D Gaussian Splatting (3DGS)
is a labor-intensive and expert-driven process, often result-
ing in inconsistent reconstructions and suboptimal results.
We propose RLGS, a plug-and-play reinforcement learn-
ing framework for adaptive hyperparameter tuning in 3DGS
through lightweight policy modules, dynamically adjusting
critical hyperparameters such as learning rates and densifica-
tion thresholds. The framework is model-agnostic and seam-
lessly integrates into existing 3DGS pipelines without ar-
chitectural modifications. We demonstrate its generalization
ability across multiple state-of-the-art 3DGS variants, includ-
ing Taming-3DGS and 3DGS-MCMC, and validate its ro-
bustness across diverse datasets. RLGS consistently enhances
rendering quality. For example, it improves Taming-3DGS by
0.7dB PSNR on the Tanks and Temple (TNT) dataset, under
a fixed Gaussian budget, and continues to yield gains even
when baseline performance saturates. Our results suggest that
RLGS provides an effective and general solution for automat-
ing hyperparameter tuning in 3DGS training, bridging a gap
in applying reinforcement learning to 3DGS.
1
Introduction
Photorealistic novel view synthesis (NVS) is a foundational
problem in computer vision and graphics, with applications
spanning immersive virtual environments and synthetic data
generation for autonomous systems. Recent advances in
neural representations have significantly improved both ren-
dering fidelity and generalization. Among NVS methods,
neural radiance fields (NeRFs) (Mildenhall et al. 2020) in-
troduced a paradigm shift by modeling scenes as continu-
ous neural functions. However, this continuous volumetric
representation incurs a substantial computational cost dur-
ing both training and inference. To address these limita-
tions, 3D Gaussian Splatting (Kerbl et al. 2023) has emerged
as a highly efficient alternative, enabling real-time render-
ing through a point-based scene representation and GPU-
friendly rasterization.
Despite its promising performance, 3DGS remains highly
sensitive to hyperparameter settings, such as learning rates
and densification thresholds. These parameters govern re-
construction quality, but selecting appropriate values re-
mains a tedious, manual, and scene-specific process. In prac-
tice, even slight mis-configurations can lead to suboptimal
rendering, overfitting, or excessive Gaussian growth, signif-
icantly limiting the accessibility and scalability of 3DGS in
real-world pipelines.
While hyperparameter optimization (HPO) is a well-
established topic in machine learning (Bergstra and Ben-
gio 2012; Snoek, Larochelle, and Adams 2012; Li et al.
2018; Jaderberg et al. 2017), most existing methods, such
as grid search, Bayesian optimization, and population-based
training, are not suited to the demands of 3DGS. These ap-
proaches typically assume access to low-cost evaluations or
differentiable objectives, and often require dozens of full
1
arXiv:2508.04078v1  [cs.GR]  6 Aug 2025

<!-- page 2 -->
training runs. This renders them impractical for computa-
tionally intensive tasks like 3DGS, where each trial can take
tens of minutes and hyperparameters interact nontrivially
with dynamic scene evolution.
We propose a novel solution: casting hyperparameter con-
trol in 3DGS as an online decision-making problem and ad-
dressing it with reinforcement learning. Specifically, we in-
troduce a policy-gradient framework that learns to dynami-
cally adjust hyperparameters, such as learning rates and den-
sification thresholds, during training of 3DGS models. Our
method treats 3DGS optimization as a Markov Decision Pro-
cess (MDP), where a lightweight policy observes training
progress and photometric error, and outputs scaling factors
for key hyperparameters.
Our approach enables fine-grained, scene-aware adapta-
tion of hyperparameters, eliminates manual tuning, and in-
tegrates seamlessly into existing 3DGS pipelines. Notably,
in scenes where state-of-the-art methods plateau despite in-
creasing the number of Gaussians, our method continues
to improve rendering quality, as illustrated in Figure 1. We
summarize our main contributions as follows:
First, we are the first to formulate hyperparameter tuning
for 3D Gaussian Splatting as a reinforcement learning prob-
lem. Our framework learns adaptive policies that dynami-
cally adjust hyperparameters during training to enhance ren-
dering quality.
Second, we introduce a lightweight, modular architecture
where separate policy agents independently control learning
rate schedules and densification behavior. This design en-
ables targeted, interpretable adaptation over different stages
of 3DGS training.
Finally, we validate RLGS across two state-of-the-art
3DGS variants: Taming-3DGS and 3DGS-MCMC, and
demonstrate consistent improvements on multiple bench-
marking datasets.
2
Related Work
We review prior work in five key areas relevant to our
method: novel view synthesis and neural representations, 3D
Gaussian Splatting, hyperparameter optimization, reinforce-
ment learning for training control, and the integration of re-
inforcement learning within 3DGS systems.
Novel View Synthesis and Neural Representations.
Novel view synthesis (NVS) aims to generate unseen views
of a scene from a set of input images. Early methods (De-
bevec, Taylor, and Malik 1996; Gortler et al. 1996; Levoy
and Hanrahan 1996; Heigl et al. 1999; Buehler et al. 2001;
Zheng et al. 2009; Kopf, Cohen, and Szeliski 2014; Cayon,
Djelouah, and Drettakis 2015; Hedman et al. 2016; Penner
and Zhang 2017; Hedman et al. 2018; Flynn et al. 2019;
Wiles et al. 2020) relied on image-based rendering tech-
niques with proxy geometry or depth maps to approximate
novel views. These approaches benefit from strong geo-
metric priors but struggle to generalize to complex scenes
with significant occlusions or view-dependent effects. Re-
cent neural approaches replace explicit geometry with learn-
able representations, such as volumetric grids (Sitzmann
et al. 2019) or implicit neural fields (Sitzmann, Zollh¨ofer,
and Wetzstein 2019), offering better generalization at the
cost of higher computational demands. Neural Radiance
Fields (NeRF) (Mildenhall et al. 2020; Barron et al. 2023)
marked a major advance by modeling the scene as a contin-
uous 5D radiance field, producing photorealistic renderings
but requiring long training and inference times.
3D
Gaussian
Splatting.
3D
Gaussian
Splatting
(3DGS) (Kerbl et al. 2023) offers a fast and scalable
alternative to NeRF by representing scenes as a set of
anisotropic 3D Gaussians, rendered using efficient ras-
terization instead of ray marching. Subsequent research
has extended 3DGS in several directions. For improved
geometry, methods such as (Huang et al. 2024; Yu, Sattler,
and Geiger 2024) refine surface fitting and better capture
fine details. For rendering quality, Mip-Splatting (Yu et al.
2024) fuses multi-resolution Gaussians to reduce aliasing.
To scale to large scenes, hierarchical (Kerbl et al. 2024) and
block-partitioned (Lin et al. 2024) structures are introduced.
Parameter compression is achieved via quantization and
sparsification (Papantonakis et al. 2024; Fan et al. 2024;
Mallick et al. 2024). Recent works also focus on rendering
improvements (Mai et al. 2024; Hou et al. 2024; Celarek
et al. 2025; Kheradmand et al. 2025) and optimization
algorithms (H¨ollein et al. 2024; Lan et al. 2025; Pehlivan
et al. 2025). Notably, Taming-3DGS (Mallick et al. 2024)
introduces error-aware pruning and learned importance
metrics to remove low-impact Gaussians and control
memory overhead without sacrificing visual fidelity. 3DGS-
MCMC (Kheradmand et al. 2024) integrates a Markov
Chain Monte Carlo (MCMC) procedure into the 3DGS
optimization loop, treating Gaussian parameters as random
variables and iteratively sampling proposals to refine the
scene representation. However, hyperparameters such as
learning rates and pruning thresholds in these variants are
still manually tuned by experts, and none of these works
offer an automated tuning solution.
Hyperparameter Optimization.
Hyperparameter opti-
mization (HPO) is crucial for achieving optimal perfor-
mance in deep learning models. Traditional methods such
as grid search and random search (Bergstra and Bengio
2012) are simple but computationally inefficient. More ad-
vanced techniques include Bayesian optimization (Snoek,
Larochelle, and Adams 2012), Hyperband (Li et al. 2018),
and population-based training (PBT) (Jaderberg et al. 2017),
which balance exploration and resource efficiency. These
approaches have been applied to tune neural architectures,
training schedules, and loss weights across various domains.
However, they are typically designed for offline, trial-based
search and are not well-suited for pipelines like 3DGS.
Reinforcement Learning for Optimization and Control.
Reinforcement learning (RL) has emerged as a power-
ful tool for automating control and optimization in ma-
chine learning pipelines. It has been applied to neural ar-
chitecture search (Zoph and Le 2017; Pham et al. 2018),
adaptive learning rate scheduling (Daniel et al. 2016;
Wu, Tucker, and Nachum 2018), and curriculum learn-
ing (Graves et al. 2017). Policy-gradient methods, such as
2

<!-- page 3 -->
REINFORCE (Williams 1992), are especially effective in
settings involving sparse or delayed rewards and are widely
used in non-differentiable or black-box optimization prob-
lems. Recent work has also explored meta-RL approaches
for learning optimizers (Andrychowicz et al. 2016; Xu,
van Hasselt, and Silver 2018). These methods have demon-
strated the potential of RL to discover dynamic training
strategies that outperform static heuristics.
Reinforcement
Learning
for
3DGS
Several
recent
works (Wang et al. 2024a,b; Wu et al. 2024) integrate 3DGS
into RL environments for downstream tasks such as robotic
control or semantic exploration. These efforts treat 3DGS
as a world model or rendering engine to support learning
agents. In contrast, we use reinforcement learning not as a
consumer of 3DGS outputs, but as a controller that directly
improves the training process of 3DGS.
To the best of our knowledge, this is the first work to ap-
ply reinforcement learning for online hyperparameter opti-
mization in 3D Gaussian Splatting. We address a key gap in
the literature by introducing a learning-based controller that
adaptively tunes hyperparameters during training to improve
rendering quality and efficiency.
3
Methodology
We treat hyperparameter tuning in 3DGS as a reinforcement
learning problem, where lightweight policy modules dynam-
ically adjust key hyperparameters during training. To this
end, we propose two lightweight policy modules—RLLR
for adaptive learning-rate scaling (policy πLR) and RLDS
for densification adjustment (policy πDS). As shown in Fig-
ure 2, these two modules are plugged into the 3DGS training
pipeline and can be integrated into other 3DGS variants.
3.1
Problem Formulation
We formulate hyperparameter tuning as a reinforcement-
learning task, using RLLR to adapt the learning rate
and RLDS to adjust densification hyperparameters when
needed. To mitigate training variance in 3DGS that hurts
policy optimization, we fold the original 30,000-step 3DGS
training schedule into J = 30,000
K
policy-phases, each span-
ning K training steps of 3DGS. We also maintain a global
iteration t to represent the training progress.
During policy-phase j, RLLR executes an inner policy
update loop with maximum length of Nlr steps. The policy
observes state sj,t as follows:
sj,t =
 ˆℓj−1, ˆτt

(1)
where ˆℓj−1 denotes the previous policy-phase’s 3DGS train-
ing loss and ˆτt encodes the current training iteration t. The
policy outputs scaling factors alr
j,t and uses them to adjust
the original learning rate hlr
orig as follows:
alr
j,t ∼πLR
 ·
 sj,t

,
hlr = hlr
orig ⊙alr
j,t
(2)
If densification is required, the RLDS module is also in-
voked using the same state input, and produces scaling fac-
tors ads
j,t for densification:
ads
j,t ∼πDS
 ·
 sj,t

,
hds = hds
orig ⊙ads
j,t.
(3)
Each inner iteration of RLLR consists of sampling learn-
ing rates, applying them for K simulated training steps,
computing the reconstruction improvement reward RLR
j,t ,
and updating the policy. Similarly, each inner iteration of
RLDS samples densification parameters, performs a densi-
fication step, trains for K simulated steps, computes RDS
j,t ,
and updates the densification policy.
The reward is defined as the improvement over the default
hyperparameters horig:
Rj,t = M(h) −M(horig)
(4)
where M(·) denotes a rendering error metric evaluated on
reward views.
After NLR inner iterations for RLLR and NDS for RLDS,
we select the best-performing configuration (hlr, hds) and
apply it to the actual update of the current policy-phase j.
Repeating this process over 30,000/K policy-phases yields
an efficient RL framework that dynamically adapts the
learning rate and densification parameters to maximize re-
construction quality while preserving the original 3DGS
pipeline.
Network Architecture
We use same network architecture
for both policies. Assume d is the number of hyperparam-
eters that need to be tuned in RLLR or RLDS. Each pol-
icy module contains a neural network to predict the distri-
bution of hyperparameters. Specifically, the network con-
sists of a GRU cell encoder (Cho et al. 2014) and a lin-
ear head for each hyperparameter. The network takes input
state sj,t as in 1 and produces residual outputs ∆µ ∈Rd
and ∆log σ ∈Rd. In addition, the module maintains two
learned base parameters, µbase ∈Rd and log σbase ∈Rd.
The final mean and log standard deviation are obtained by
adding the network-predicted residuals to these learnable
bases:
µ = µbase + ∆µ,
log σ = log σbase + ∆log σ.
(5)
Action Sampling and Exploration
An action a ∈Rd is
sampled from the resulting Gaussian distribution:
a ∼N
 µ, σ2
,
(6)
where σ = exp(log σ). To encourage exploration, we add
an entropy bonus to the policy objective (Mnih et al. 2016;
Schulman et al. 2017). We train the policy by minimizing
the negative reward-weighted log-probability of actions and
an entropy term.
3.2
Reward Design
Simulated Training
The effect of a sampled hyperparam-
eter action a appears only after several training iterations.
After sampling a hyperparameter action a, we simulate the
future K training iterations to compute the reward as Equa-
tion 4.
Reward View Sampling
To obtain a reward signal that ac-
curately captures multi-view improvements induced by ac-
tions (i.e, changing learning rate or densification hyperpa-
rameter), we withhold reward views from the training set.
3

<!-- page 4 -->
W + ΔW1
Begin
RLLR 
RLDS
End
Need DS
j ≥ 
30,000
𝐾𝐾
Yes
Yes
(a) Main loop
(b) RLLR and RLDS and modules
(c) Weight updating and densification
hyperparameters h by sampled actions
3DGS weights W
Hyperparameter actions distributions
3DGS weights updating ΔW ,  larger 
bar means larger weight updating
πLR
Simulated 
Training
i ≥ NLR
Reward
Computing
Set LR
W + ΔW2
Yes
i = 1
πDS
DS Step
Simulated
Training
i ≥ NDS
Reward
Computing
Yes
i = 1
i + 1
i + 1
j + 1, t + K
j = 1, t = 1
i = 1
i = NLR
i = 1
i = NDS
Sj,t
Sj,t
hlr
hds
Figure 2: Overview of the proposed framework. (a) Our method follows the vanilla 3DGS training loop with two plug-and-
play modules, RLLR and RLDS, for adjusting the learning rate and densification hyperparameters, respectively. (b) This sub-
figure presents our RLLR and RLDS modules. At policy phase j with global iteration t, the two policies take as input the state
sj,t, which encodes the number of completed training iterations and the previous phase’s loss (see Equation 1). Each policy then
samples an action, alr
j,t and ads
j,t, respectively, to generate the hyperparameters hlr and hds. Both policies are optimized via policy
gradient to maximize reconstruction improvement after simulated training. (c) This sub-figure illustrates how hyperparameters
hlr and hds control updates to the model weights W and affect densification, respectively.
(a)
Training views
Reward views
Sampling views
(b)
(c)
Repeat sampling
Possible views
Splitting views
Figure 3: (a) Initial training views. (b) Split the initial set into
training and reward views, then sample training views from
the current pool of possible views. (c) After each Ishuffle op-
erations, resample the reward views.
We find that including reward views in the training set in-
troduces a bias toward training views. At the same time,
per-scene 3D reconstruction benefits from utilizing as many
training views as possible. Balancing between leveraging ev-
ery view and avoiding over-fitting on bias is non-trivial.
To address this problem, as shown in Figure 3, we split
the initial training views into the reward views and remain-
ing training views. Every Ishuffle iterations, we keep the re-
ward views fixed for short-term reward stability. To lever-
age all views over the long term, we randomly shuffle and
re-split the initial training views into reward views and re-
maining training views. This design is inspired by cross val-
idation. However, unlike traditional cross-validation, which
partitions data for evaluating a fixed hyperparameter con-
figuration, our approach uses the reward views as online
feedback to the policy network. The policy thus learns to
generate hyperparameters on the fly, directly influencing the
3DGS training process rather than merely evaluating it.
3.3
3DGS Backbone Design
We propose a plug-and-play reinforcement learning frame-
work for tuning 3DGS hyperparameters. However, changing
hyperparameters inevitably alters the final number of Gaus-
sians, introducing additional challenges for performance as-
sessment. Therefore, we integrate our method into two state-
of-the-art 3DGS backbones (TMGS (Mallick et al. 2024)
and 3DGS-MCMC (Kheradmand et al. 2024)) that can set
final number of Gaussians.
4
Experiments
We begin with an overview of datasets and evaluation met-
rics in Section 4.1. Next, we compare our approach with hy-
perparameter search methods on public benchmark datasets
in Section 4.2. Then, we validate our method’s effectiveness
on a large-scale real-world dataset in Section 4.3. In Sec-
tion 4.4, we apply our method to another Gaussian splatting
backbone to demonstrate its adaptation. Section 4.5 presents
an ablation study of our method’s key components. Finally,
implementation details can be found in Section 4.6.
4.1
Datasets and Metric Settings
TNT and Deep Blending.
Due to the high computational
cost of baselines such as random search, we conduct ex-
periments on two real-world datasets: the Tanks and Tem-
ples dataset (TNT) (Knapitsch et al. 2017) and the Deep
Blending dataset (Hedman et al. 2018). Each dataset con-
tains two scenes. These datasets include bounded indoor and
4

<!-- page 5 -->
Method
num of trials↓
TNT
Deep Blending
Average
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
TMGS
1
24.04
0.851
0.170
30.14
0.907
0.235
27.09
0.879
0.203
RS + TMGS
64
24.15
0.860
0.162
29.43
0.905
0.244
26.79
0.883
0.203
BO + TMGS
64
24.48
0.861
0.162
29.59
0.906
0.242
27.04
0.884
0.202
Ours + TMGS
1
24.74
0.866
0.158
30.26
0.911
0.233
27.50
0.889
0.195
Table 1: Comparison of different methods on the TNT (Knapitsch et al. 2017) and Deep Blending datasets (Hedman et al.
2018). The last three columns show the average of PSNR, SSIM, and LPIPS across both datasets.
TMGS + Random
TMGS + BO
TMGS + Ours
TMGS
GT
Figure 4: Qualitative comparisons on the TNT and Deep Blending Datasets.
unbounded outdoor environments with rich background de-
tails.
DL3DV-140.
To support large-scale tuning across di-
verse real-world scenes, we evaluate on the DL3DV-140
dataset (Ling et al. 2024), which consists of 140 scenes with
varied content, created for the novel view synthesis task.
Mip-NeRF360.
To evaluate the generality of our method,
we additionally test on the Mip-NeRF360 dataset (Barron
et al. 2022). Following the protocol of (Kheradmand et al.
2024), we select seven representative scenes and downsam-
ple the test images for evaluation.
Train/test split.
For all the datasets, we follow prior
work (Kerbl et al. 2023; Mallick et al. 2024) and reserve
every 8th image for testing.
Metrics.
We report PSNR, SSIM, and LPIPS (Zhang et al.
2018) following standard practice in 3DGS papers (Kerbl
et al. 2023; Mallick et al. 2024).
4.2
Comparison with Hyperparameter Tuning
Methods
In this experiment, we evaluate our policy–gradient tuner
against two standard hyper-parameter optimization (HPO)
techniques. We use the Taming-3DGS as the backbone.
Baselines.
Random Search (RS) (Bergstra and Ben-
gio 2012) samples hyperparameters uniformly at random.
Bayesian Optimisation (BO) (Bergstra et al. 2011): we em-
ploy the Tree-structured Parzen Estimator (TPE) sampler
from Optuna (Akiba et al. 2019), which optimizes the same
training-PSNR objective. All baselines are built on the latest
official codebase of TMGS (Mallick et al. 2024).All search
5

<!-- page 6 -->
Method
PSNR↑
SSIM↑
LPIPS↓
TMGS
29.75
0.909
0.131
TMGS + Ours
29.96
0.912
0.125
Table
2:
Evaluation
on
the
large-scale
DL3DV-140
dataset (Ling et al. 2024).
TMGS + Ours
TMGS
GT
Figure 5: Qualitative comparisons on the DL3DV Datasets.
methods operate over the same hyperparameter ranges. For
Bayesian Optimization, we provide the official hyperparam-
eter as a good informative prior.
Results.
As shown in Table 1, our experiments on two
benchmark datasets demonstrate that the default TMGS
hyperparameters deliver strong performance for the two
dataset. Despite conducting 64 trials, baseline methods ran-
dom search (RS) and Bayesian optimization (BO) fail to out-
perform these defaults on the Deep Blending dataset. In con-
trast, our reinforcement learning–based tuner consistently
surpasses them. On the TNT dataset, our method can en-
hance TMGS by a large margin (0.7dB). Visual compar-
isons are shown in Figure 4. In the last example shown in
the figure, our method successfully preserves the small light
bulb, whereas search-based methods fail to retain it.
4.3
Generalization to Real-World Large-Scale
Dataset
This experiment evaluates our method on DL3DV, a large-
scale, user-captured dataset, to assess its adaptability to
complex and diverse real-world data domains.
Baseline.
As established in the previous experiment,
TMGS is a strong baseline. Therefore, we adopt the state-
of-the-art TMGS method as our baseline and integrate our
Method
PSNR↑
SSIM↑
LPIPS↓
3DGS-MCMC
29.89
0.900
0.190
3DGS-MCMC + Ours
30.12
0.897
0.144
Table 3: Adaption to 3DGS-MCMC on the MipNeRF 360
dataset (Barron et al. 2022).
3DGS-MCMC + Ours
3DGS-MCMC
GT
Figure 6: Qualitative comparisons on the MipNeRF 360
Datasets.
approach into it to evaluate how it can be improved on the
DL3DV-140 benchmark.
Results.
Table 2 summarizes quantitative comparisons be-
tween TMGS baseline and our adaptive tuning on the
DL3DV-140 benchmark. On average across all 140 scenes,
our method achieves a 0.21 dB gain in PSNR, and a 0.006 re-
duction in LPIPS compared to TMGS with its default hyper-
parameters. These improvements consistently demonstrate
the effectiveness of our method over the TMGS backbone.
Figure 5 provides qualitative examples: while the baseline
sometimes blurs thin structures and misses fine textures, our
method preserves these details.
4.4
Adaptation to 3DGS-MCMC Backbone
Baseline.
We also adapt our method to another 3D Gaus-
sian approach to demonstrate its practical value, especially
as new 3DGS methods are emerging rapidly. Specifically,
we apply our method to 3DGS-MCMC (Kheradmand et al.
2024), a state-of-the-art technique that also allows control
over the number of Gaussians. Since 3DGS-MCMC’s den-
sification process does not involve any tunable hyperpa-
rameters—except for the opacity threshold, whose modifi-
cation would compromise the training process—we replace
the densification hyperparameter with the regularization hy-
perparameter recommended by the authors. This adaptation
further demonstrates the flexibility of our framework, which
is not limited to tuning the learning rate and densification
parameters.
Results.
Table 3 presents quantitative comparisons be-
tween the 3DGS-MCMC baseline and our adaptively tuned
6

<!-- page 7 -->
version. Our method yields a 0.23 dB gain in PSNR and a
0.046 (24 %) reduction in LPIPS. Qualitative results in Fig-
ure 6 further illustrate that our approach better preserves thin
structures and fine textures; in the last example, only our
method accurately captures the light reflections on the chair.
These results demonstrate the generalizability of our frame-
work on other 3DGS variants.
4.5
Ablation Study
We perform an ablation study on each component of the pro-
posed framework using the TNT dataset (Table 4). We begin
with the two main modules, followed by a detailed analy-
sis of specific design choices, including the GRU encoder,
entropy bonus, loss input, and reward-view sampling.
Baseline TMGS.
”w/o RLLR and RLDS” is the baseline
TMGS without our modification. This causes a 0.7 dB PSNR
drop, showing that our two modules can significantly en-
hance reconstruction quality.
Learning rate hyperparameters.
“w/o RLLR” replaces
our reinforcement learning–based learning rate policy with
the original fixed schedule. This variant suffers a 0.41 dB
drop in PSNR, along with corresponding degradations in
SSIM and LPIPS, indicating that our RLLR module plays
a significant role in the overall performance improvement.
Densification hyperparameters.
“w/o RLDS” replaces
our densification hyperparameters policy with the original
one in TMGS. Omitting RLDS yields a 0.09 dB PSNR re-
duction, indicating the effectiveness of our RLDS module.
Recurrent module architecture.
“w/o GRU” replaces the
GRU encoder with linear layers. This change incurs a 0.11
dB PSNR loss, confirming that GRU better captures long-
term information than linear layers.
Entropy term in reward computing.
“w/o Entropy” re-
moves the entropy term from the policy reward, discourag-
ing exploration. This leads to a 0.12 dB PSNR decrease, sug-
gesting that the entropy bonus helps the policy avoid subop-
timal behaviors.
Loss input.
“w/o Loss Input” excludes the previous-phase
reconstruction loss from the policy’s state input. This causes
a 0.09 dB PSNR drop, highlighting that feeding back loss
information stabilizes policy updates and improves decision
making.
Reward view sampling.
“w/o Reward sampling” uses a
reward set from each phase’s training set instead of sam-
pling from a hold reward set. This incurs a 0.11 dB PSNR
loss, demonstrating that diversity in reward sampling aids
generalization across viewpoints.
Overall, both the policy modules (RLLR and RLDS) and
the design choices (GRU encoder, entropy bonus, loss in-
put, and reward-view sampling) contribute significantly to
the improved rendering quality.
4.6
Implementation Details
To stabilize policy training, we clip gradients to a maximum
norm of 2.4. The reward-set length is set to 2, and each pol-
icy phase spans K = 20 training steps. We empirically set
Method
PSNR↑
SSIM↑
LPIPS↓
Ours
24.74
0.866
0.158
w/o RLLR and RLDS
24.04
0.851
0.170
w/o RLLR
24.33
0.861
0.161
w/o RLDS
24.65
0.863
0.158
w/o GRU
24.63
0.864
0.159
w/o Entropy
24.62
0.863
0.162
w/o Loss input
24.65
0.864
0.160
w/o Reward sampling
24.63
0.863
0.161
Table 4: Ablation study on the TNT dataset.
Ishuffle to 1000: smaller values destabilize the reward sig-
nal, while larger values reduce view diversity over time. The
learning rate for our policy networks is fixed at 1×10−4. The
Train scene takes 25 minutes to complete training using our
integrated TMGS model on an NVIDIA A6000 Ada GPU.
The learning-rate policy module RLLR controls five hyper-
parameters related to 3DGS optimization: position, scaling,
rotation, opacity, and the base spherical harmonic feature
coefficients. The densification policy module RLDS adjusts
two key hyperparameters: the density threshold and the ex-
ternal scaling factor used during densification. We do not
modify the opacity threshold in the densification process, as
changing it would significantly alter the number of Gaus-
sians and potentially break the program. Please refer to the
supplemental material for additional implementation details
and extended results.
5
Conclusions
In this work, we propose a reinforcement learning–based
framework (RLGS) for adjusting hyperparameters during
3DGS training. By formulating hyperparameter tuning as an
online decision-making problem, our plug-and-play method
uses lightweight policies that observe photometric error and
training progress to dynamically adjust learning rates and
densification hyperparameters.
RLGS integrates seamlessly into existing 3DGS pipelines
without requiring architectural changes. Extensive experi-
ments demonstrate its effectiveness: on the TNT dataset,
RLGS improves PSNR by up to 0.7 dB. On the large-
scale DL3DV-140 dataset, it delivers consistent improve-
ments across diverse real-world scenes and achieves better
qualitative results. Furthermore, we show that RLGS gener-
alizes well to 3DGS variants, including Taming-3DGS and
3DGS-MCMC.
A current limitation is that the learned policy network
functions as a black box, making it difficult to interpret deci-
sions. Future work will explore more interpretable architec-
tures, incorporate hybrid reinforcement learning techniques,
and broaden the scope of adaptive control to other aspects of
3DGS optimization.
References
Akiba, T.; Sano, S.; Yanase, T.; Ohta, T.; and Koyama, M.
2019. Optuna: A next-generation hyperparameter optimiza-
7

<!-- page 8 -->
tion framework. In Proceedings of the 25th ACM SIGKDD
international conference on knowledge discovery & data
mining, 2623–2631.
Andrychowicz, M.; Denil, M.; Gomez, S.; Hoffman, M. W.;
Pfau, D.; Schaul, T.; Shillingford, B.; and De Freitas, N.
2016. Learning to learn by gradient descent by gradient de-
scent. Advances in neural information processing systems,
29.
Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.;
and Hedman, P. 2022.
Mip-nerf 360: Unbounded anti-
aliased neural radiance fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 5470–5479.
Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.;
and Hedman, P. 2023. Zip-nerf: Anti-aliased grid-based neu-
ral radiance fields. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, 19697–19705.
Bergstra, J.; Bardenet, R.; Bengio, Y.; and K´egl, B. 2011.
Algorithms for Hyper-Parameter Optimization. In Advances
in Neural Information Processing Systems.
Bergstra, J.; and Bengio, Y. 2012. Random search for hyper-
parameter optimization. The journal of machine learning
research, 13(1): 281–305.
Buehler, C.; Bosse, M.; McMillan, L.; Gortler, S.; and
Cohen, M. 2001.
Unstructured Lumigraph Rendering.
In Proceedings of the 28th Annual Conference on Com-
puter Graphics and Interactive Techniques, SIGGRAPH
’01, 425–432. New York, NY, USA: Association for Com-
puting Machinery. ISBN 158113374X.
Cayon, R. O.; Djelouah, A.; and Drettakis, G. 2015.
A
bayesian approach for selective image-based rendering us-
ing superpixels. In 2015 International Conference on 3D
Vision, 469–477. IEEE.
Celarek, A.; Kopanas, G.; Drettakis, G.; Wimmer, M.; and
Kerbl, B. 2025. Does 3D Gaussian Splatting Need Accu-
rate Volumetric Rendering? In Computer Graphics Forum,
e70032. Wiley Online Library.
Cho, K.; Van Merri¨enboer, B.; Bahdanau, D.; and Ben-
gio, Y. 2014.
On the properties of neural machine
translation: Encoder-decoder approaches.
arXiv preprint
arXiv:1409.1259.
Daniel, C.; Taylor, J.; Nowozin, S.; and Blundell, C. 2016.
Learning step size controllers for robust neural network
training. In Proceedings of the AAAI Conference on Arti-
ficial Intelligence.
Debevec, P. E.; Taylor, C. J.; and Malik, J. 1996. Model-
ing and Rendering Architecture from Photographs: A Hy-
brid Geometry- and Image-Based Approach. In Proceedings
of the 23rd Annual Conference on Computer Graphics and
Interactive Techniques, SIGGRAPH ’96, 11–20. New York,
NY, USA: Association for Computing Machinery.
ISBN
0897917464.
Fan, Z.; Wang, K.; Wen, K.; Zhu, Z.; Xu, D.; Wang, Z.; et al.
2024. Lightgaussian: Unbounded 3d gaussian compression
with 15x reduction and 200+ fps. Advances in neural infor-
mation processing systems, 37: 140138–140158.
Flynn, J.; Broxton, M.; Debevec, P.; DuVall, M.; Fyffe, G.;
Overbeck, R.; Snavely, N.; and Tucker, R. 2019. Deepview:
View synthesis with learned gradient descent. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2367–2376.
Gortler, S. J.; Grzeszczuk, R.; Szeliski, R.; and Cohen, M. F.
1996. The lumigraph. In Siggraph, volume 96, 43–54.
Graves, A.; Bellemare, M. G.; Menick, J.; Munos, R.; and
Kavukcuoglu, K. 2017. Automated curriculum learning for
neural networks. arXiv preprint arXiv:1704.03003.
Hedman, P.; Philip, J.; Price, T.; Frahm, J.-M.; Drettakis, G.;
and Brostow, G. 2018. Deep blending for free-viewpoint
image-based rendering. In SIGGRAPH Asia 2018 Technical
Papers, 257. ACM.
Hedman, P.; Ritschel, T.; Drettakis, G.; and Brostow, G.
2016. Scalable Inside-out Image-Based Rendering. ACM
Trans. Graph., 35(6).
Heigl, B.; Koch, R.; Pollefeys, M.; Denzler, J.; and Gool, L.
J. V. 1999. Plenoptic Modeling and Rendering from Image
Sequences Taken by Hand-Held Camera. In Mustererken-
nung 1999, 21. DAGM-Symposium, 94–101. Berlin, Heidel-
berg: Springer-Verlag. ISBN 3540663819.
H¨ollein, L.; Boˇziˇc, A.; Zollh¨ofer, M.; and Nießner, M.
2024. 3dgs-lm: Faster gaussian-splatting optimization with
levenberg-marquardt. arXiv preprint arXiv:2409.12892.
Hou, Q.; Rauwendaal, R.; Li, Z.; Le, H.; Farhadzadeh, F.;
Porikli, F.; Bourd, A.; and Said, A. 2024. Sort-free Gaus-
sian Splatting via Weighted Sum Rendering. arXiv preprint
arXiv:2410.18931.
Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024.
2D Gaussian Splatting for Geometrically Accurate Radiance
Fields. In SIGGRAPH 2024 Conference Papers. Association
for Computing Machinery.
Jaderberg, M.; Dalibard, V.; Osindero, S.; Czarnecki, W. M.;
Donahue, J.; Razavi, A.; Vinyals, O.; Green, T.; Dunning,
I.; Simonyan, K.; et al. 2017. Population based training of
neural networks. arXiv preprint arXiv:1711.09846.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian Splatting for Real-Time Radiance Field
Rendering. ACM Transactions on Graphics, 42(4).
Kerbl, B.; Meuleman, A.; Kopanas, G.; Wimmer, M.; Lan-
vin, A.; and Drettakis, G. 2024. A hierarchical 3d gaussian
representation for real-time rendering of very large datasets.
ACM Transactions on Graphics (TOG), 43(4): 1–15.
8

<!-- page 9 -->
Kheradmand, S.; Rebain, D.; Sharma, G.; Sun, W.; Tseng,
Y.-C.; Isack, H.; Kar, A.; Tagliasacchi, A.; and Yi, K. M.
2024. 3d gaussian splatting as markov chain monte carlo.
Advances in Neural Information Processing Systems, 37:
80965–80986.
Kheradmand, S.; Vicini, D.; Kopanas, G.; Lagun, D.; Yi,
K. M.; Matthews, M.; and Tagliasacchi, A. 2025. Stochastic-
Splats: Stochastic Rasterization for Sorting-Free 3D Gaus-
sian Splatting. arXiv preprint arXiv:2503.24366.
Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017.
Tanks and temples: Benchmarking large-scale scene recon-
struction. ACM Transactions on Graphics (ToG), 36(4): 1–
13.
Kopf, J.; Cohen, M. F.; and Szeliski, R. 2014. First-person
hyper-lapse videos. ACM Transactions on Graphics (TOG),
33(4): 78.
Lan, L.; Shao, T.; Lu, Z.; Zhang, Y.; Jiang, C.; and Yang, Y.
2025. 3DGS2: Near Second-order Converging 3D Gaussian
Splatting. arXiv:2501.13975.
Levoy, M.; and Hanrahan, P. 1996. Light field rendering.
In Proceedings of the 23rd annual conference on Computer
graphics and interactive techniques, 31–42. ACM.
Li, L.; Jamieson, K.; DeSalvo, G.; Rostamizadeh, A.; and
Talwalkar, A. 2018. Hyperband: A novel bandit-based ap-
proach to hyperparameter optimization. Journal of Machine
Learning Research, 18(185): 1–52.
Lin, J.; Li, Z.; Tang, X.; Liu, J.; Liu, S.; Liu, J.; Lu, Y.;
Wu, X.; Xu, S.; Yan, Y.; et al. 2024. Vastgaussian: Vast 3d
gaussians for large scene reconstruction. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 5166–5175.
Ling, L.; Sheng, Y.; Tu, Z.; Zhao, W.; Xin, C.; Wan, K.;
Yu, L.; Guo, Q.; Yu, Z.; Lu, Y.; et al. 2024. Dl3dv-10k: A
large-scale scene dataset for deep learning-based 3d vision.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 22160–22169.
Mai, A.; Hedman, P.; Kopanas, G.; Verbin, D.; Futschik, D.;
Xu, Q.; Kuester, F.; Barron, J. T.; and Zhang, Y. 2024. Ever:
Exact volumetric ellipsoid rendering for real-time view syn-
thesis. arXiv preprint arXiv:2410.01804.
Mallick, S. S.; Goel, R.; Kerbl, B.; Steinberger, M.; Car-
rasco, F. V.; and De La Torre, F. 2024.
Taming 3dgs:
High-quality radiance fields with limited resources. In SIG-
GRAPH Asia 2024 Conference Papers, 1–11.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing
Scenes as Neural Radiance Fields for View Synthesis. In
ECCV.
Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.;
Harley, T.; Silver, D.; and Kavukcuoglu, K. 2016. Asyn-
chronous Methods for Deep Reinforcement Learning.
In
Proceedings of the 33rd International Conference on Ma-
chine Learning (ICML).
Papantonakis, P.; Kopanas, G.; Kerbl, B.; Lanvin, A.; and
Drettakis, G. 2024. Reducing the memory footprint of 3d
gaussian splatting. Proceedings of the ACM on Computer
Graphics and Interactive Techniques, 7(1): 1–17.
Pehlivan, H.; Camiletto, A. B.; Foo, L. G.; Habermann,
M.; and Theobalt, C. 2025. Second-order Optimization of
Gaussian Splats with Importance Sampling. arXiv preprint
arXiv:2504.12905.
Penner, E.; and Zhang, L. 2017.
Soft 3d reconstruction
for view synthesis. ACM Transactions on Graphics (TOG),
36(6): 1–11.
Pham, H.; Guan, M. Y.; Zoph, B.; Le, Q. V.; and Dean, J.
2018. Efficient Neural Architecture Search via Parameter
Sharing. In Proceedings of the International Conference on
Machine Learning (ICML).
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and
Klimov, O. 2017. Proximal Policy Optimization Algorithms.
arXiv preprint arXiv:1707.06347.
Sitzmann, V.; Thies, J.; Heide, F.; Nießner, M.; Wetzstein,
G.; and Zollhofer, M. 2019. Deepvoxels: Learning persis-
tent 3d feature embeddings.
In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
2437–2446.
Sitzmann, V.; Zollh¨ofer, M.; and Wetzstein, G. 2019.
Scene representation networks: Continuous 3d-structure-
aware neural scene representations. Advances in Neural In-
formation Processing Systems, 32.
Snoek, J.; Larochelle, H.; and Adams, R. P. 2012. Practi-
cal bayesian optimization of machine learning algorithms.
Advances in neural information processing systems, 25.
Wang, J.; Zhang, Q.; Sun, J.; Cao, J.; Han, G.; Zhao, W.;
Zhang, W.; Shao, Y.; Guo, Y.; and Xu, R. 2024a. Reinforce-
ment learning with generalizable gaussian splatting. In 2024
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), 435–441. IEEE.
Wang, J.; Zhang, Z.; Zhang, Q.; Li, J.; Sun, J.; Sun, M.; He,
J.; and Xu, R. 2024b. Query-based semantic gaussian field
for scene representation in reinforcement learning.
arXiv
preprint arXiv:2406.02370.
Wiles, O.; Gkioxari, G.; Szeliski, R.; and Johnson, J. 2020.
Synsin: End-to-end view synthesis from a single image. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, 7467–7477.
Williams, R. J. 1992. Simple statistical gradient-following
algorithms for connectionist reinforcement learning.
Ma-
chine Learning, 8(3-4): 229–256.
9

<!-- page 10 -->
Wu, Y.; Pan, L.; Wu, W.; Wang, G.; Miao, Y.; Xu, F.;
and Wang, H. 2024.
Rl-gsbridge: 3d gaussian splatting
based real2sim2real method for robotic manipulation learn-
ing. arXiv preprint arXiv:2409.20291.
Wu, Y.; Tucker, G.; and Nachum, O. 2018.
Variance re-
duction for reinforcement learning in input-driven environ-
ments. In Advances in Neural Information Processing Sys-
tems.
Xu, Z.; van Hasselt, H.; and Silver, D. 2018. Meta-gradient
reinforcement learning. arXiv preprint arXiv:1805.09801.
Yu, Z.; Chen, A.; Huang, B.; Sattler, T.; and Geiger, A. 2024.
Mip-Splatting: Alias-free 3D Gaussian Splatting. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 19447–19456.
Yu, Z.; Sattler, T.; and Geiger, A. 2024.
Gaussian opac-
ity fields: Efficient adaptive surface reconstruction in un-
bounded scenes.
ACM Transactions on Graphics (TOG),
43(6): 1–13.
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In Proceedings of the IEEE conference
on computer vision and pattern recognition, 586–595.
Zheng, K. C.; Colburn, A.; Agarwala, A.; Agrawala, M.;
Salesin, D.; Curless, B.; and Cohen, M. F. 2009.
Paral-
lax photography: creating 3d cinematic effects from stills.
In Proceedings of Graphics Interface 2009, 111–118. Cana-
dian Information Processing Society.
Zoph, B.; and Le, Q. V. 2017. Neural Architecture Search
with Reinforcement Learning. In Proceedings of the Inter-
national Conference on Learning Representations (ICLR).
Zwicker, M.; Pfister, H.; Van Baar, J.; and Gross, M. 2001.
EWA volume splatting. In Proceedings Visualization, 2001.
VIS’01., 29–538. IEEE.
10

<!-- page 11 -->
A
Overview
In the supplementary material, we provide:
• The 3DGS preliminary in Appendix B.
• More qualitative comparisons in Appendix C.
• Per-scene quantitative results in Appendix D.
B
3DGS Preliminary
Given multi-view captured images with poses, 3D Gaus-
sian Splatting (Kerbl et al. 2023) (3DGS) learns a set of
anisotropic 3D Gaussians by minimizing a rendering loss
through differentiable rasterization. In 3DGS (Kerbl et al.
2023), each Gaussian i is defined by its center µi, covariance
Σi, opacity σi, and spherical harmonics coefficients hi. The
opacity contributed by Gaussian i at an arbitrary point x is
given by:
αi = σi exp

−1
2(x −µi)T Σ−1
i (x −µi)

.
(7)
Each gaussian’s covariance Σi is positive semi-definite and
we factor it into a rotation matrix Ri and a diagonal scale
matrix Si:
Σi = RiSiST
i RT
i .
(8)
To rasterize a 2D image from a given viewpoint, 3D Gaus-
sian splatting approximate the perspective projection of each
3D Gaussian into the image plane (Zwicker et al. 2001).
Specifcially, a 3D Gaussian with parameters (µi, Σi) is ap-
proximated by a 2D Gaussian with mean µ2D
i
and covari-
ance Σ2D
i
. Given the world-to-camera extrinsic W and the
intrinsic matrix K, these are computed as
µ2D
i
= (K((Wµi)/(Wµi)z))1:2,
(9)
Σ2D
i
= (JWΣiW T JT )1:2,1:2,
(10)
where J is the Jacobian of the projective transformation.
The subscript (·)1:2 selects the x and y components of the
projected mean, while (·)1:2,1:2 extracts the 2D spatial co-
variance in the image plane. After sorting the Gaussians in
increasing depth order for ordered volumetric rendering, a
pixel’s color is computed as:
I =
X
i∈N
ci α2D
i
Y
j<i
 1 −α2D
j

,
(11)
where α2D
i
denotes the 2D variant of Eq. (7), modified by re-
placing µi, Σi, x with µ2D
i
, Σ2D
i
, x2D (the pixel coordinate).
The term ci denotes the RGB color produced by evaluating
the spherical harmonics with the view direction (from Gaus-
sian means µi to camera centers) and coefficients.
C
Additional Qualitative Comparisons
In this section, we present additional qualitative compar-
isons to demonstrate the visual improvements achieved by
our method. Figure 7 shows results on the TNT and Deep
Blending Datasets. We observe that our method produces
noticeably better visual quality in many scenes from the
Deep Blending dataset. For example, in the fourth row, our
method correctly preserves the edge detail in the brightly
lit ceiling, which is not captured as well by the baselines.
Figure 8 shows results on the MipNeRF 360 Dataset. When
integrated with our method, 3DGS-MCMC preserves better
details. In the second-to-last row, which contains challeng-
ing blurriness, our method produces results that are nearly
indistinguishable from the ground truth. Figure 9 shows re-
sults on the DL3DV Dataset, demonstrating consistent im-
provements in line with the previous two figures. In the first
row of Figure 9, where many small spots are affected by
ground reflections, our method can reconstruct the rich tex-
tures.
D
Per-scene Quantitative Results
Table 5 shows per-scene quantitative results produced by
TMGS (Mallick et al. 2024) integrated with our method
on the TNT (Knapitsch et al. 2017) Dataset. Table 6 shows
per-scene quantitative results produced by TMGS (Mallick
et al. 2024) integrated with our method on the Deep Blend-
ing Dataset (Hedman et al. 2018). Table 7 shows per-scene
quantitative results produced by 3DGS integrated with our
method. For per-scene results on the DL3DV dataset (Ling
et al. 2024), please refer to the attached dl3dv.txt file.
Scene
PSNR
SSIM
LPIPS
Count (M)
train
23.24
0.837
0.191
1.09
truck
26.25
0.895
0.124
2.58
Table 5: Per-scene Quantitative results on the TNT
Dataset (Knapitsch et al. 2017) using TMGS (Mallick et al.
2024) enhanced with our method.
Scene
PSNR
SSIM
LPIPS
Count (M)
drjohnson
29.88
0.910
0.232
3.27
playroom
30.65
0.913
0.234
2.33
Table 6: Per-scene quantitative results on the Deep Blending
Dataset (Hedman et al. 2018) using TMGS (Mallick et al.
2024) enhanced with our method.
Scene
PSNR
SSIM
LPIPS
Count (M)
bicycle
26.30
0.810
0.163
5.90
garden
28.36
0.887
0.087
5.20
stump
27.60
0.818
0.166
4.75
room
32.99
0.940
0.168
1.50
counter
29.70
0.927
0.159
1.20
kitchen
32.64
0.941
0.105
1.80
bonsai
33.26
0.955
0.160
1.30
Table 7: Per-scene quantitative results on the MipNeRF
360 (Barron et al. 2022) dataset using 3DGS-MCMC (Kher-
admand et al. 2024) enhanced with our method.
11

<!-- page 12 -->
TMGS + Random
TMGS + BO
TMGS + Ours
TMGS
GT
Figure 7: Qualitative comparisons on the TNT and Deep Blending Datasets.
12

<!-- page 13 -->
3DGS-MCMC + Ours
3DGS-MCMC
GT
Figure 8: Qualitative comparisons on the MipNeRF 360 Dataset.
13

<!-- page 14 -->
TMGS + Ours
TMGS
GT
Figure 9: Qualitative comparisons on the DL3DV Dataset.
14
