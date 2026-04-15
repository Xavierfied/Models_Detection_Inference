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