# Inference Detection

A unified Inference Comparison system focused on MediaPipe and ControlNet open-pose landmark models, with RetinaFace face detection.

## Features

- **mp_runner**      — face & pose landmark detection via MediaPipe Landmarker. Best for single person detection and live\recorded feed.
- **cn_op_runner**   — face & pose landmark detection via ControlNet's OpenPose detector. Best for crowd detection and recorded feed.
- **rf_runner**      — Robust Face Detection and landmark matching via RetinaFace. Best for huge crowds and recorded feed due to power hungry.
- **yv8pose_runner** — Pose Detection and Identifies if the person is sitting or standing.
- Supports images, video files, and webcam input
- Results saved to `results/` as `{filename}_{detector}.{ext}`

## Setup

1. **Create and activate a virtual environment with python <=3.11.x**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# ControlNet OpenPose detection
python main.py --model cn_op --save  --source "samples/band.mp4"

# RetinaFace detection
python main.py --model rf --save  --source "samples/band.mp4"

# MediaPipe
python main.py --model mp --view  --source 0

# MediaPipe
python main.py --model yv8pose --view  --source 0

```

## Parameters

```
 Parameter     Default    Description 

 `--source`     `Required`    Image/video path or webcam index (e.g. `0`) 
 `--model`      `Required`    choices=["mp", "rf", "cn_op"]
 `--save`       `None`        Save in "results"
 `--view`       `None`        Views while processing
 `--output_dir` `results`     output folder
```

## Project Structure

```
main.py              
utils/
   args.py              - Running Params
   helpers.py           - For Using Runners
runners/
   cn_op_runner.py      - ControlNet Open-pose
   mp_runner.py         - Media pipe
   rf_runner.py         - Retina Face
   yv8pose_runner.py    - Yolov8 Pose Estimation
weights/
  face_landmarker.task
  pose_landmarker_full.task
```

## Output

Results are saved in the `results/` directory as `{filename}_{model}_{time_string}.{ext}`, e.g.:

MediaPipe `.task` model files are to be stored in `weights/` and downloaded them via the following link and store them in "weights directory"
```
https://drive.google.com/drive/folders/1j1qoZIK7e7ecVOeXEGK5yrAQcIDEilf5?usp=sharing
```


## Update Logs:

- **15/04/2026:**
```
      - Added Yolov8 Pose detection
      - Identifies if the person is sitting or standing via following criterion:
            - Aspect Ratio, Standing person would have a ratio of 2:1 as compared to 1.2:1 for sitting
            - Hips and Knees placement, checking the horizontal and vertical travel between the two kpts
```
