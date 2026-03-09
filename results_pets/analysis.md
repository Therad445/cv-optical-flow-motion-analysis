# Analysis

## Lucas-Kanade

- Mean number of active points: 179.96
- Mean number of lost points per frame: 0.01
- Mean displacement: 0.64

Observations:
- Lucas–Kanade gives interpretable trajectories and works well on textured regions.
- The tracker becomes less stable in weakly textured areas and under motion blur.
- Points may disappear during fast motion, occlusions and abrupt illumination changes.

## Farneback

- Mean flow magnitude: 0.27
- Mean motion area ratio: 0.0361

Observations:
- Farnebäck captures motion densely across the frame.
- It is suitable for obtaining motion masks and highlighting moving regions.
- The method may produce noisy motion estimates near shadows, blur and low-contrast boundaries.

## Comparison

- Lucas–Kanade is better for feature-level trajectory analysis.
- Farnebäck is better for dense motion estimation and segmentation-like motion masks.
- In real applications, the choice depends on whether the task requires sparse stable tracks or dense motion fields.
