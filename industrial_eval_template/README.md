# Industrial Evaluation Template

Use this directory to standardize benchmark outputs.

## Structure

```text
industrial_eval_template/
  method_example/
    run_config.json
    scene_001_metal_corridor/
      condition_003_lowtexture_dark_dynamic/
        route_01_loop/
          tracking/
          mapping/
          rendering/
          summary.json
```

## Required Files

- `tracking/estimated_trajectory.tum`
- `tracking/tracking_status.csv`
- `mapping/reconstruction_meta.json`
- `summary.json`

