# Industrial Problem Formulation Draft

## 1. Task Definition

We consider an embodied agent equipped with calibrated multimodal sensors operating in a complex industrial environment that contains three dominant degradation factors: texture scarcity, low and spatially varying illumination, and dynamic interference from moving objects or agents. Given a time-ordered sensor stream

\[
\mathcal{O} = \{o_t\}_{t=1}^{T},
\]

the goal is to estimate a camera or robot trajectory

\[
\hat{\mathcal{X}} = \{\hat{\mathbf{T}}_t\}_{t=1}^{T},
\]

and, optionally, a scene representation

\[
\hat{\mathcal{M}},
\]

that supports geometric reconstruction, dense mapping, or view synthesis.

The benchmark is intended to cover both classical SLAM systems and recent 3D Gaussian Splatting based SLAM or mapping methods. Accordingly, the formulation separates tracking quality, mapping quality, and robustness to environmental degradation.

## 2. Observation Model

At time step \(t\), the system receives a synchronized observation

\[
o_t = \{I_t, D_t, \mathbf{u}_t, L_t, m_t\},
\]

where \(I_t\) denotes the RGB image, \(D_t\) denotes depth or range observation when available, \(\mathbf{u}_t\) denotes inertial measurement, \(L_t\) denotes lighting-related metadata such as measured lux or image brightness statistics, and \(m_t\) denotes optional dynamic-region annotations or masks.

Not all methods are required to consume every modality. However, all methods are evaluated against the same condition-aware benchmark protocol.

## 3. Environmental Condition Variables

Each sequence is associated with a condition tuple

\[
c = (\tau, \lambda, \delta),
\]

where:

- \(\tau\) denotes texture level
- \(\lambda\) denotes illumination level
- \(\delta\) denotes dynamic level

In the proposed benchmark, these three variables are treated as first-class benchmark factors rather than informal scene descriptions.

Texture level is intended to capture the availability of stable local appearance cues. In practice, it can be approximated using image-gradient statistics, Laplacian variance, or detected feature density. Illumination level captures low-light severity and photometric non-uniformity using structured metadata such as average lux, minimum lux, brightness distribution, dark-pixel ratio, and saturation ratio. Dynamic level captures the amount of non-static content, which may be represented through dynamic-mask coverage, dynamic-object count, or scripted event labels.

## 4. State Estimation Objective

For the tracking component, a method estimates a pose sequence

\[
\hat{\mathbf{T}}_t \in SE(3),
\]

which should approximate ground-truth poses

\[
\mathbf{T}^{*}_t.
\]

Tracking quality is evaluated through pose error and operational stability. Let

\[
E_{\text{track}} = \mathcal{L}_{\text{ATE}} + \mathcal{L}_{\text{RPE}} + \mathcal{L}_{\text{fail}},
\]

where \(\mathcal{L}_{\text{ATE}}\) measures global trajectory deviation, \(\mathcal{L}_{\text{RPE}}\) measures local relative drift, and \(\mathcal{L}_{\text{fail}}\) summarizes failure behavior such as tracking-loss count, lost duration, relocalization failure, or loop-closure failure.

The benchmark does not assume that all methods optimize this exact loss internally. Instead, it defines the evaluation target that all methods must satisfy.

## 5. Mapping Objective

For the mapping component, a method outputs a scene representation

\[
\hat{\mathcal{M}},
\]

which may take the form of a point cloud, mesh, TSDF, neural field, or Gaussian representation. This representation is expected to support one or more of the following:

- geometric reconstruction
- dense map generation
- novel-view rendering
- downstream localization or navigation

Mapping quality is evaluated against a reference scene representation \(\mathcal{M}^{*}\) through geometric metrics such as Chamfer distance, accuracy, and completeness. When the method supports rendering, optional photometric metrics such as PSNR, SSIM, and LPIPS may also be reported.

We write the mapping objective abstractly as

\[
E_{\text{map}} = \mathcal{L}_{\text{geom}}(\hat{\mathcal{M}}, \mathcal{M}^{*}) + \mathcal{L}_{\text{photo}}(\hat{\mathcal{M}}, \mathcal{M}^{*}),
\]

where the photometric term is optional and benchmark-track dependent.

## 6. Robustness Objective Under Complex Degradation

Industrial deployment requires more than average-case accuracy. A method should remain stable as texture decreases, illumination worsens, and dynamic interference increases. We therefore define a condition-aware robustness objective

\[
R(\tau, \lambda, \delta),
\]

which summarizes method behavior under each condition tuple. In practice, this quantity is instantiated through grouped metrics rather than a single scalar. Examples include:

- ATE under low-texture sequences
- track-loss rate under dark sequences
- reconstruction completeness under low-texture and dark sequences
- dynamic contamination ratio under medium or high dynamic sequences

This condition-aware view is central to the benchmark. Two methods with similar global averages may behave very differently once evaluated under controlled industrial degradation factors.

## 7. Dynamic Contamination Formulation

For dynamic scenes, a static-world assumption may cause moving objects to be fused into the estimated map. Let \(\Omega^{\text{dyn}}_t\) denote the dynamic region at frame \(t\). The benchmark measures the degree to which dynamic content is incorporated into the final map representation using a dynamic contamination ratio

\[
\rho_{\text{dyn}} = \frac{\text{dynamic content fused into } \hat{\mathcal{M}}}{\text{total reconstructed content in evaluated region}}.
\]

This quantity may be approximated through dynamic masks, temporal inconsistency checks, or reference-map comparison depending on available annotations. The exact implementation can vary, but the reporting target remains the same: methods should distinguish persistent structure from transient motion.

## 8. Benchmark Output Requirements

A valid benchmark submission may include three levels of output:

1. tracking output: estimated trajectory and tracking-status logs
2. mapping output: reconstruction artifact and mapping metadata
3. robustness output: condition-grouped summaries and failure statistics

Not every method is required to support every benchmark track. However, any reported result must provide the artifacts required by the corresponding track specification.

## 9. Short Version For Methods Section

We formulate industrial complex-environment SLAM as a condition-aware state-estimation and mapping problem under three coupled degradations: texture scarcity, photometric degradation, and dynamic interference. Given synchronized multimodal observations over time, a method estimates the sensor trajectory and, optionally, a dense scene representation for reconstruction or rendering. Each sequence is associated with structured condition metadata describing texture, illumination, and dynamic severity. Evaluation is performed along three axes: tracking quality, mapping quality, and robustness under degradation. Tracking quality is measured by ATE, RPE, and failure behavior such as track loss and relocalization. Mapping quality is measured by geometric consistency metrics, and optionally photometric rendering metrics. Robustness is measured through grouped performance under controlled condition tuples and through dynamic contamination analysis in non-static scenes.

## 10. Short Version For Problem Formulation Section

Let \(\mathcal{O} = \{o_t\}_{t=1}^{T}\) denote the synchronized observation stream collected in an industrial environment. The objective is to recover a pose sequence \(\hat{\mathcal{X}} = \{\hat{\mathbf{T}}_t\}_{t=1}^{T}\) and optionally a scene representation \(\hat{\mathcal{M}}\). Each sequence is labeled by a condition tuple \(c=(\tau,\lambda,\delta)\), where \(\tau\), \(\lambda\), and \(\delta\) denote texture level, illumination level, and dynamic level, respectively. Performance is evaluated not only by global trajectory and reconstruction accuracy, but also by grouped robustness under these degradation factors, including tracking failure statistics and dynamic contamination in the estimated map.
