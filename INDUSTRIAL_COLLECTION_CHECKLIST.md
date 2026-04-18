# Industrial Dataset Collection Checklist

## Before Capture

- confirm target task: tracking / mapping / robustness
- assign scene id and condition id
- calibrate camera intrinsics
- calibrate cross-sensor extrinsics
- verify timestamp synchronization
- prepare lux meter
- prepare trajectory ground-truth system
- prepare operator route sheet
- prepare dynamic-event script if controlled dynamic capture is used

## Per Sequence

- record scene id
- record condition id
- record route id
- record operator id
- record sensor rig id
- record start time
- measure average lux
- measure minimum lux
- note texture level
- note dynamic level
- note reflective / glass / smoke / dust presence
- confirm calibration file version

## Capture Protocol

- begin with 5 to 10 seconds static sensor hold
- walk planned route once at normal speed
- repeat same route if dynamic protocol is needed
- include at least one turn-heavy segment
- include one stationary observation window
- if loop route, clearly close the loop

## Dynamic Protocol

- log dynamic object type
- log count
- log approximate speed
- log intended path
- keep at least one repeatable scripted version

## After Capture

- verify RGB frames are complete
- verify depth / LiDAR stream completeness
- verify IMU completeness
- verify timestamps align
- export metadata.json
- export ground-truth trajectory
- generate quick preview video
- note obvious corruption or dropped frames

## Release Gate

Do not release a sequence if:

- timestamps are broken
- calibration is missing
- GT trajectory is missing
- condition label is unclear
- file naming is inconsistent
- dynamic condition is not documented

