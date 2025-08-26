import argparse
import json
from ultralytics import YOLO
from ultralytics.utils import TQDM
from ultralytics.utils.torch_utils import get_num_params, get_flops

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on VisDrone dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/VisDrone.yaml', 
                       help='Path to the dataset YAML file (default: ultralytics/cfg/datasets/VisDrone.yaml)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size for the model (default: 640)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for inference (default: 16)')
    parser.add_argument('--split', type=str, default='test', 
                       help='Dataset split to use for evaluation (train, val, test) (default: test)')
    parser.add_argument('--conf', type=float, default=0.001, 
                       help='Object confidence threshold for detection (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS (default: 0.6)')
    parser.add_argument('--device', type=str, default='', help='Device to run evaluation on (cpu, cuda, etc.)')
    parser.add_argument('--verbose', action='store_true', help='Display per-class metrics')
    parser.add_argument('--save-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    return parser.parse_args()

class TQDMCallback:
    """Custom callback to display progress during model validation"""
    
    def __init__(self):
        self.pbar = None
        self.total_steps = 0
        
    def on_val_start(self, validator):
        """Initialize progress bar at start of validation"""
        if hasattr(validator, 'dataloader'):
            self.total_steps = len(validator.dataloader)
            self.pbar = TQDM(total=self.total_steps, desc="Validation Progress")
    
    def on_val_batch_end(self, validator):
        """Update progress bar after each batch"""
        if self.pbar:
            self.pbar.update(1)
    
    def on_val_end(self, validator):
        """Close progress bar at end of validation"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

def main():
    args = parse_args()

    # Load the trained model
    model = YOLO(args.weights)
    
    # Setup progress callback - always enabled
    tqdm_callback = TQDMCallback()
    model.add_callback("on_val_start", tqdm_callback.on_val_start)
    model.add_callback("on_val_batch_end", tqdm_callback.on_val_batch_end)
    model.add_callback("on_val_end", tqdm_callback.on_val_end)

    # Validate the model on the dataset
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device
    )

    # Get model information (parameters and FLOPs)
    params = get_num_params(model.model)
    flops = get_flops(model.model, args.imgsz)

    # Print detailed metrics
    print("\n" + "="*90)
    print("YOLO Model Evaluation Results on VisDrone Dataset")
    print("="*90)
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Model: {args.weights}")
    print(f"Image Size: {args.imgsz}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: {args.iou}")
    print("-"*90)
    
    # Extract and display key metrics
    if hasattr(results, 'box'):
        print("Overall Detection Metrics:")
        print(f"  mAP@0.5:       {results.box.map50:.4f} (mean Average Precision at IoU=0.5)")
        print(f"  mAP@0.5:0.95:  {results.box.map:.4f} (mean Average Precision at IoU=0.5:0.95)")
        print(f"  Precision:     {results.box.mp:.4f} (overall precision)")
        print(f"  Recall:        {results.box.mr:.4f} (overall recall)")
        print(f"  F1-Score:      {results.box.f1.mean():.4f} (harmonic mean of precision and recall)")
        
        # Per-class metrics if requested
        if args.verbose and hasattr(results, 'names') and len(results.names) > 1:
            print("\nPer-Class mAP@0.5 Metrics:")
            for i, class_name in results.names.items():
                if i < len(results.box.ap50):
                    print(f"  {class_name:15}: {results.box.ap50[i]:.4f}")
        
        # Additional metrics
        print("\nAdditional Metrics:")
        print(f"  Number of Classes: {len(results.names) if hasattr(results, 'names') else 'N/A'}")
        print(f"  mP:            {results.box.mp:.4f} (mean precision across classes)")
        print(f"  mR:            {results.box.mr:.4f} (mean recall across classes)")
        
    else:
        # For cases where results is a dict or other format
        print(f"Results: {results}")
    
    # Display speed metrics if available
    if hasattr(results, 'speed'):
        print("\nInference Speed Metrics:")
        total_speed = sum(results.speed.values())
        print(f"  Preprocess:    {results.speed['preprocess']:.2f}ms per image")
        print(f"  Inference:     {results.speed['inference']:.2f}ms per image")
        print(f"  Postprocess:   {results.speed['postprocess']:.2f}ms per image")
        print(f"  Total:         {total_speed:.2f}ms per image")
        print(f"  FPS:           {1000/total_speed:.2f} frames per second")
    
    # Display model information
    print("\nModel Information:")
    print(f"  Parameters:    {params:,} ({params/1e6:.2f} M)")
    if flops > 0:
        print(f"  GFLOPs:        {flops:.2f} GFLOPs")
    else:
        print("  GFLOPs:        N/A")
    
    print("="*90 + "\n")

    # Create save directory if it doesn't exist
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save results in multiple formats
    # 1. Save as text file
    results_txt_path = os.path.join(args.save_dir, f"evaluation_results_{args.split}.txt")
    with open(results_txt_path, 'w') as f:
        f.write("YOLO Model Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.weights}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Image Size: {args.imgsz}\n")
        f.write(f"Confidence Threshold: {args.conf}\n")
        f.write(f"IoU Threshold: {args.iou}\n")
        f.write("="*50 + "\n")
        
        if hasattr(results, 'box'):
            f.write("Overall Detection Metrics:\n")
            f.write(f"  mAP@0.5:       {results.box.map50:.4f}\n")
            f.write(f"  mAP@0.5:0.95:  {results.box.map:.4f}\n")
            f.write(f"  Precision:     {results.box.mp:.4f}\n")
            f.write(f"  Recall:        {results.box.mr:.4f}\n")
            f.write(f"  F1-Score:      {results.box.f1.mean():.4f}\n")
            
            # Per-class metrics if available
            if hasattr(results, 'names') and len(results.names) > 1:
                f.write("\nPer-Class mAP@0.5 Metrics:\n")
                for i, class_name in results.names.items():
                    if i < len(results.box.ap50):
                        f.write(f"  {class_name:15}: {results.box.ap50[i]:.4f}\n")
            
            # Additional metrics
            f.write("\nAdditional Metrics:\n")
            f.write(f"  Number of Classes: {len(results.names) if hasattr(results, 'names') else 'N/A'}\n")
            f.write(f"  mP:            {results.box.mp:.4f}\n")
            f.write(f"  mR:            {results.box.mr:.4f}\n")
        
        # Speed metrics if available
        if hasattr(results, 'speed'):
            f.write("\nInference Speed Metrics:\n")
            total_speed = sum(results.speed.values())
            f.write(f"  Preprocess:    {results.speed['preprocess']:.2f}ms per image\n")
            f.write(f"  Inference:     {results.speed['inference']:.2f}ms per image\n")
            f.write(f"  Postprocess:   {results.speed['postprocess']:.2f}ms per image\n")
            f.write(f"  Total:         {total_speed:.2f}ms per image\n")
            f.write(f"  FPS:           {1000/total_speed:.2f} frames per second\n")
        
        # Model information
        f.write("\nModel Information:\n")
        f.write(f"  Parameters:    {params:,} ({params/1e6:.2f} M)\n")
        if flops > 0:
            f.write(f"  GFLOPs:        {flops:.2f} GFLOPs\n")
        else:
            f.write("  GFLOPs:        N/A\n")
    
    print(f"Evaluation results saved in '{args.save_dir}' directory:")
    print(f"  - Text format: {results_txt_path}")
    #print(f"  - JSON format: {results_json_path}")

if __name__ == '__main__':
    main()