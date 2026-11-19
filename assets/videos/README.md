# CARLA Driving Videos

Dashcam clips rendered in the CARLA simulator, bundled with the application as sample input for lane detection. Each clip is 15 seconds of a forward-facing camera on a vehicle driven by CARLA's Traffic Manager, with the vehicle moving for the whole clip.

| File | Town | Scene | Weather | Episode | Source frames |
|---|---|---|---|---|---|
| `carla_town10hd_urban_clear_ep04.mp4` | Town10HD | Urban, marked lanes | ClearNoon | 4 | 2400 to 2699 |
| `carla_town10hd_urban_clear_ep09.mp4` | Town10HD | Urban, marked lanes | ClearNoon | 9 | 5400 to 5699 |
| `carla_town10hd_urban_rain_sunset_ep19.mp4` | Town10HD | Urban, wet road, low light | MidRainSunset | 19 | 11400 to 11699 |
| `carla_town04_highway_wet_sunset_ep18.mp4` | Town04 | Highway | WetSunset | 18 | 10800 to 11099 |

## Format

- H.264 Constrained Baseline, level 3.0, `yuv420p`, no audio, `faststart` for streaming playback on Android.
- 640 x 384 at 20 frames per second, 300 frames per clip.
- Captured at 320 x 192 and upscaled 2x with Lanczos resampling. No detail beyond the capture resolution is present.
- Camera: pinhole, 90 degree horizontal field of view; at capture resolution fx = fy = 160, cx = 160, cy = 96. Double these values for the bundled resolution.

## Provenance

Frames are taken unchanged from the `vehicle_town10hd` and `vehicle_town04` environments of the [mini-carla-192x320](https://huggingface.co/datasets/kamwoh/mini-carla-192x320) dataset, rendered with CARLA 0.9.16 at the `Low` quality preset. Clips were selected for continuous motion and visible lane markings; the Town04 and Town06 clear-weather highway episodes were left out because the `Low` preset washes out their markings.

## Licence

The source dataset is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CARLA's code is MIT and its assets, including the towns and vehicles shown, are released under CC BY. These clips are redistributed under CC BY 4.0, separately from the MIT licence that covers the code in this repository.

Credit: [CARLA simulator](https://carla.org/), Dosovitskiy et al., *CARLA: An Open Urban Driving Simulator*, CoRL 2017; video frames from mini-carla-192x320 by kamwoh.
