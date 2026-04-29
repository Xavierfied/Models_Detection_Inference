import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Inference Comparison Tool")

    parser.add_argument("--source",
                         type=str,
                         required=True,
                         help="Image path, video path, or webcam index '0'")

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=["mp", "rf", "cn_op", "yv8pose"],
                        help="Inference backend")

    parser.add_argument("--save",
                        action="store_true",
                        help="Save the output")

    parser.add_argument("--output_dir",
                        type=str,
                        default="results",
                        help="Folder to save results")

    parser.add_argument("--view", action="store_true", help="Display the window")

    return parser.parse_args()


def get_lift_args():
    parser = argparse.ArgumentParser(description="LiftOptimize — squat & deadlift form checker")

    parser.add_argument("--source",
                        type=str,
                        default="0",
                        help="Video path or 0 for webcam (default: 0)")

    parser.add_argument("--runner",
                        type=str,
                        default="mp",
                        choices=["mp", "yv8"],
                        help="Pose backend: mp (MediaPipe) or yv8 (YOLOv8)")

    parser.add_argument("--weights",
                        type=str,
                        default="weights/yolov8s-pose.pt",
                        help="Path to YOLOv8 pose weights (only used with --runner yv8)")

    parser.add_argument("--save",
                        action="store_true",
                        help="Save output video to results/")

    return parser.parse_args()


def get_train_args():
    parser = argparse.ArgumentParser(description="Train the ActivityLSTM classifier")

    parser.add_argument("--data_dir",
                        type=str,
                        default="training_data",
                        help="Root folder with standing/, walking/, lifting/ subfolders")

    parser.add_argument("--save_path",
                        type=str,
                        default="weights/activity_lstm.pt",
                        help="Where to write the trained weights")

    parser.add_argument("--epochs",
                        type=int,
                        default=30)

    parser.add_argument("--batch",
                        type=int,
                        default=32)

    parser.add_argument("--lr",
                        type=float,
                        default=1e-3)

    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="cuda or cpu (default: auto-detect)")

    return parser.parse_args()