# Industrial Complex Environment Dataset Spec

## Goal

Build a validation dataset for industrial SLAM / 3DGS-SLAM systems under three target degradations:

1. textureless surfaces
2. low-light conditions
3. dynamic environments

The dataset is meant to answer:

1. when tracking starts to drift in texture-poor scenes
2. when reconstruction quality collapses in low light
3. when dynamic objects contaminate mapping
4. which methods fail first under combined degradation

## Scope

This dataset is not a generic industrial video archive.

It is a controlled benchmark with:

- repeatable trajectories
- condition labels
- trajectory ground truth
- scene reconstruction reference
- dynamic-region annotations

## Dataset Structure

```text
industrial-complex-dataset/
  README.md
  splits/
    train.txt
    val.txt
    test.txt
  calibration/
    sensor_rig_01/
      intrinsics/
      extrinsics/
      time_sync.json
  scenes/
    scene_001_metal_corridor/
      condition_001_lowtexture_normal_static/
        rgb/
        depth/
        imu/
        lidar/
        ground_truth/
        masks_dynamic/
        calibration_link.txt
        metadata.json
      condition_002_lowtexture_dark_static/
      condition_003_lowtexture_dark_dynamic/
    scene_002_storage_night/
    scene_003_equipment_room/
```

## Scene Design

Start with 4 core scenes:

1. metal corridor
2. storage aisle at night
3. equipment room passage
4. AGV / forklift traffic corridor

Recommended expansion:

5. pipe gallery
6. white-wall inspection corridor
7. machine workshop with repetitive structure
8. mixed glass / reflective control room

## Condition Matrix

Each scene should be captured under a controlled subset of:

- texture level: `high`, `medium`, `low`, `extreme_low`
- illumination level: `normal`, `dim`, `dark`, `extreme_dark`, `mixed_hard_shadow`
- dynamic level: `static`, `low_dynamic`, `medium_dynamic`, `high_dynamic`

Recommended first-pass combinations:

1. `low texture + normal light + static`
2. `low texture + dark + static`
3. `low texture + dark + medium dynamic`
4. `low texture + extreme dark + high dynamic`
5. `repetitive structure + dark + dynamic occlusion`

## Sensor Requirements

Minimum rig:

- RGB camera
- depth camera or LiDAR
- IMU
- synchronized timestamps
- calibrated intrinsics / extrinsics

Preferred rig:

- stereo or RGB-D camera
- IMU
- 3D LiDAR
- lux meter
- high-accuracy reference system for ground truth

Ground truth options:

1. motion capture
2. total station
3. laser scanner plus offline registration
4. pre-built high-accuracy map plus pose alignment

## Sequence Types

For each condition, collect at least 4 trajectory types:

1. slow straight traversal
2. out-and-back loop closure route
3. turn-heavy route
4. occlusion-interrupted route

Recommended operator motion profile:

- walking speed: 0.3 to 1.2 m/s
- turning pauses included
- both smooth and abrupt rotation segments

## Annotation Requirements

Priority order:

1. camera trajectory ground truth
2. dense reference geometry or reference point cloud
3. dynamic-region masks
4. illumination metadata
5. texture metadata

Optional:

- semantic labels
- object tracks
- instance masks for dynamic agents

## Benchmark Tasks

### 1. Industrial-Track

Tracking and localization robustness only.

Metrics:

- ATE
- RPE
- tracking failure count
- relocalization success rate
- loop closure success rate

### 2. Industrial-Map

Dense mapping and reconstruction quality.

Metrics:

- Chamfer distance
- completeness
- geometric accuracy
- PSNR
- SSIM
- LPIPS

### 3. Industrial-Robust

Failure boundary under compound degradation.

Metrics:

- success rate by condition
- drift against texture level
- drift against lux
- dynamic contamination ratio
- map corruption in dynamic regions

## Condition Metadata

Every sequence must include:

- scene id
- condition id
- operator id
- sensor rig id
- capture date
- route length
- motion type
- loop or non-loop
- texture level
- illumination level
- dynamic level
- average lux
- minimum lux
- dynamic agent types
- reflective / transparent / smoke flags

## Quantifying Textureless Conditions

Do not label texture subjectively only.

Store:

- Laplacian variance
- average image gradient magnitude
- feature count per frame
- repeat-pattern indicator

Suggested summary buckets:

- `high`: feature-rich
- `medium`: minor degradation
- `low`: sparse stable features
- `extreme_low`: near-featureless

## Quantifying Low-Light Conditions

Store:

- average lux
- minimum lux
- image mean brightness
- dark-pixel ratio
- saturated-pixel ratio
- estimated image SNR if available

Suggested summary buckets:

- `normal`: >100 lux
- `dim`: 30-100 lux
- `dark`: 5-30 lux
- `extreme_dark`: <5 lux

Adjust thresholds if your target site has different baseline illumination.

## Quantifying Dynamic Complexity

Store:

- dynamic object count
- dynamic object classes
- approximate speed bucket
- frame-wise dynamic mask coverage ratio
- longest continuous occlusion duration

Suggested levels:

- `static`
- `low_dynamic`
- `medium_dynamic`
- `high_dynamic`

## Controlled Dynamic Protocol

Use repeatable dynamic events first:

1. one pedestrian crossing
2. two-way pedestrian crossing
3. AGV pass-through
4. forklift pass-through
5. door opening / closing
6. moving machinery or conveyor motion

Keep uncontrolled activity as a separate split or tag.

## Splits

Recommended split logic:

- `train`: simpler and medium conditions
- `val`: mixed conditions, known scenes
- `test`: hardest conditions and held-out scenes

Do not leak:

- same trajectory with only minor temporal offset
- same scene-condition pair across train and test
- same ground-truth map with nearly identical motion path across splits

## MVP Version

Start with:

- 4 scenes
- 3 conditions per scene
- 4 routes per condition
- total: 48 sequences

If that is too large, start from:

- 3 scenes
- 2 conditions per scene
- 4 routes per condition
- total: 24 sequences

## Deliverables

A usable release must contain:

1. raw sensor data
2. calibration
3. ground-truth trajectories
4. dynamic masks
5. metadata
6. benchmark definitions
7. evaluation scripts
8. baseline result tables

## Recommended First Scenes

### Scene 1: Metal Corridor

- long low-texture walls
- repeated structure
- loop route possible

### Scene 2: Storage Aisle Night

- dark shelves
- local lighting only
- strong shadows and reflective packaging

### Scene 3: Equipment Room Passage

- narrow traversal
- cables, cabinets, weak texture
- human occlusion likely

### Scene 4: AGV / Forklift Corridor

- strong dynamic interference
- repeated industrial layout
- partial occlusion and motion blur

