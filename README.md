android-opencv-lanedetection
============================

On-device lane detection for Android with OpenCV, revived as the perception front end for **Lifting SPDE-GMRF Fields: GPU Estimation and On-Device Forward Filtering**.

## Lineage

| Period            | Status                                                                                              |
|-------------------|------------------------------------------------------------------------------------------------------|
| 2014              | Original application developed by Wahib-ul-Haq as an MSc Computer Science semester project at TUM   |
| Summer 2019       | Derived into an advanced driver assistance systems (ADAS) venture                                   |
| October 2024      | ADAS venture incorporated                                                                            |
| April 2025        | Venture disbanded; codebase refactored                                                               |
| 17 November 2026  | Revived, following tacit approval, as part of the SPDE-GMRF study                                    |

## Motivation

Researchers in ergonomics and human factors need inexpensive ways to observe driver behaviour across different scenarios. The original application showed that a phone mounted on the dashboard can detect lane markings and estimate the Standard Deviation of Lane Position (SDLP), a common measure of lateral control and driver impairment.

Lane position alone, however, does not explain why a driver's control changes. A driver moves between a small number of behavioural modes, such as calm, alert, high stress and panic, and drifts gradually within each mode. The revived project treats lane position as one observation stream for a model that separates those two effects.

## Objectives

1. Retain the on-device lane detection and SDLP estimation as the perception stage.
2. Emit lane position as a compact, time-stamped observation stream.
3. Feed that stream into a hidden Markov model (HMM) whose continuous drift is modelled by a sparse Gaussian field (SPDE-GMRF).
4. Keep the heavy estimation on a GPU and the per-sample regime decision on the device.

## Assumptions

- Lane markings are visible to a forward-facing camera for most of each drive.
- SDLP over a short window is an informative, though incomplete, signal of the current behavioural regime.
- Perception may run in near time; the regime filter requires only the latest available summary.
- Simulated and recorded drives are used for model design. Results on public roads require separate validation.

## Architecture

```
 Phone camera (dashboard)
        |
        v
 +-------------------------------+
 | OpenCV lane detection (NDK)   |   grayscale, edges, Hough lines,
 | red overlay, lane position    |   lane fit and tracking
 +---------------+---------------+
                 |  lateral offset, SDLP window
                 v
 +-------------------------------+        periodic         +------------------------------+
 | On-device forward filter      | <---------------------- | GPU estimation               |
 | int8 HMM over four regimes    |   A, field offsets      | sparse Q(theta), Cholesky,   |
 | argmax = current mode         |                         | kappa and tau estimation     |
 +-------------------------------+                         +------------------------------+
```

The device runs one small recursion per sample. The GPU re-estimates the transition matrix and the field offsets periodically, away from the latency path.

## Status

The lane detection and SDLP estimation are inherited from the original project. Integration with the SPDE-GMRF model is in progress. No accuracy or latency figures have been measured for the revived pipeline, and none are claimed.

## Attribution

- Original application and project report, *Computer Vision on Nomadic Devices for Human Factors Engineering*, by Wahib-ul-Haq (TUM): [project report](https://www.academia.edu/11329425/Computer_vision_on_nomadic_devices_for_human_factors_engineering).
- Core lane detection based on [lane-detection](https://github.com/jdorweiler/lane-detection) by J. Dorweiler, described in [Revisiting lane detection using OpenCV](http://www.transistor.io/revisiting-lane-detection-using-opencv.html), ported to the Android NDK and optimised for smartphones in the original application.
- Related case study: [Lifting SPDE-GMRF Fields: GPU Estimation and On-Device Forward Filtering](https://whatheheckisthis.github.io/case-studies/spde-gmrf-gpu-edge/).

## Licence

MIT. The original copyright notice (Copyright (c) 2014 wahib-ul-haq) is retained in [LICENSE](LICENSE).
