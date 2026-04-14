# Inference Detection

A unified Inference Comparison system focused on MediaPipe and ControlNet open-pose landmark models, with RetinaFace face detection.

## Features

- **Face** — face landmark detection via MediaPipe Face Landmarker
- **Hands** — hand landmark detection via MediaPipe
- **Pose** — pose landmark detection via MediaPipe
- **RetinaFace** — face detection and landmarks via RetinaFace
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
python main.py --model mp --save  --source 0

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
   rf_runner.py         - rf_runner.py
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


