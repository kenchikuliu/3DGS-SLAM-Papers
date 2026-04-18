# Industrial Dataset Template

Use this directory as the starting point for real capture.

## Contents

- `scenes/`: sequence layout
- `splits/`: train / val / test lists
- `calibration/`: sensor rig calibration
- `templates/`: metadata and logging templates

## First Recommended Scenes

1. `scene_001_metal_corridor`
2. `scene_002_storage_night`
3. `scene_003_equipment_room`
4. `scene_004_agv_corridor`

## First Recommended Conditions

1. `condition_001_lowtexture_normal_static`
2. `condition_002_lowtexture_dark_static`
3. `condition_003_lowtexture_dark_dynamic`

## Usage

1. copy one scene-condition folder for each new sequence
2. fill in `templates/metadata.example.json`
3. rename it to `metadata.json`
4. fill in `templates/capture_log.csv`
5. store all calibration files under `calibration/<sensor_rig_id>/`

