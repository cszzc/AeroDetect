
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model="/root/autodl-tmp/ultralytics-mainPro/ultralytics/cfg/models/11/AFPN.yaml", task = 'detect')
    model.train(data="/root/autodl-tmp/ultralytics-mainPro/ultralytics/cfg/datasets/AAA_my_datasets.yaml",
                imgsz=640,
                epochs=100,
                batch=8,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                amp=True,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
