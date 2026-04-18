# Industrial Dataset Evaluation Spec

## Goal

Define a fixed evaluation interface for industrial SLAM / 3DGS-SLAM benchmarking under:

1. textureless conditions
2. low-light conditions
3. dynamic environments

This spec is designed so different methods can be evaluated with the same input and output contract.

## Benchmark Tracks

### 1. Industrial-Track

Tracking and localization only.

Required outputs:

- estimated trajectory
- tracking status timeline
- relocalization events

Core metrics:

- ATE RMSE
- ATE mean
- RPE translation RMSE
- RPE rotation RMSE
- track loss count
- total lost duration
- relocalization success count
- loop closure success count

### 2. Industrial-Map

Dense mapping / reconstruction quality.

Required outputs:

- reconstructed point cloud or mesh
- optional rendered views

Core metrics:

- Chamfer distance
- accuracy
- completeness
- precision at threshold
- recall at threshold

Optional view-synthesis metrics:

- PSNR
- SSIM
- LPIPS

### 3. Industrial-Robust

Robustness against combined degradation.

Required outputs:

- trajectory
- reconstruction
- dynamic-region handling result

Core metrics:

- success rate by condition
- drift by texture level
- drift by illumination level
- drift by dynamic level
- dynamic contamination ratio
- map corruption ratio in masked dynamic regions

## Input Contract

Each evaluated sequence is identified by:

- `scene_id`
- `condition_id`
- `route_id`

Method runners should consume:

- sensor data from the sequence directory
- calibration from `calibration/<sensor_rig_id>/`
- sequence metadata from `metadata.json`

## Output Directory Layout

```text
eval_runs/
  <method_name>/
    run_config.json
    scene_001_metal_corridor/
      condition_003_lowtexture_dark_dynamic/
        route_01_loop/
          tracking/
            estimated_trajectory.tum
            tracking_status.csv
            relocalization_events.csv
          mapping/
            reconstruction.ply
            reconstruction_meta.json
          rendering/
            metrics.json
          summary.json
```

## Required Output Files

### Tracking

`estimated_trajectory.tum`

- format: `timestamp tx ty tz qx qy qz qw`
- timestamps in seconds
- poses in world coordinates

`tracking_status.csv`

Columns:

- `timestamp`
- `frame_id`
- `tracking_state`
- `is_lost`
- `is_keyframe`

Allowed `tracking_state` values:

- `tracking`
- `lost`
- `relocalized`
- `initializing`

`relocalization_events.csv`

Columns:

- `timestamp_start`
- `timestamp_end`
- `success`
- `notes`

### Mapping

`reconstruction.ply`

- fused point cloud or extracted surface

`reconstruction_meta.json`

Fields:

- `representation_type`
- `point_count`
- `triangle_count`
- `has_color`
- `coordinate_frame`

### Summary

`summary.json`

Fields:

- `scene_id`
- `condition_id`
- `route_id`
- `method_name`
- `runtime_sec`
- `peak_vram_mb`
- `peak_ram_mb`
- `success`
- `notes`

## Ground Truth Inputs

### Trajectory GT

Preferred file:

- `ground_truth/trajectory.tum`

### Reconstruction Reference

Preferred files:

- `ground_truth/reference_cloud.ply`
- `ground_truth/reference_mesh.ply`

### Dynamic Masks

Preferred directory:

- `masks_dynamic/`

Suggested per-frame mask names:

- `000001.png`
- `000002.png`

## Condition-Aware Reporting

Each leaderboard row should include:

- scene
- condition
- route
- texture level
- illumination level
- dynamic level

This is mandatory. Do not report only global averages.

## Standard Aggregations

### Per Sequence

Compute metrics for every route independently.

### Per Condition

Aggregate over all routes with same:

- scene
- condition

### Per Degradation Level

Aggregate over:

- all low-texture sequences
- all dark sequences
- all medium/high dynamic sequences

### Global

Report full average only after the grouped metrics.

## Failure Definition

A run is considered failed if any of the following happens:

1. no valid trajectory output
2. more than 50 percent sequence lost
3. reconstruction file missing for mapping track
4. output timestamps cannot be aligned with GT

## Reporting Tables

Recommended tables:

1. tracking table
   - ATE
   - RPE
   - loss count
   - relocalization success

2. mapping table
   - Chamfer
   - accuracy
   - completeness
   - runtime

3. robustness table
   - success rate by texture / light / dynamic level
   - dynamic contamination ratio

## Minimum Baselines

For a meaningful first benchmark, evaluate at least:

1. a classical SLAM baseline
2. a neural implicit SLAM baseline
3. a 3DGS-SLAM baseline
4. your target industrial method

## First Release Rule

Do not publish leaderboard numbers until:

- at least 3 scenes are complete
- each scene has at least 2 conditions
- each condition has at least 2 routes
- GT and metadata are complete

