import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size for the model')  # 修改为 imgsz
    parser.add_argument('--batch', type=int, default=16, help='Batch size for inference')  # 修改为 batch
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load the trained model
    model = YOLO(args.weights)

    # Validate the model on the dataset
    results = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)  # 修改为 imgsz 和 batch

    # Print the mAP and other metrics
    print(f"Validation Results: {results}")
