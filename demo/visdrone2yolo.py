import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# 用于转换 VisDrone 的边界框到 YOLO 的 xywh 格式
def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    # 计算中心坐标和宽高，并归一化
    center_x = (box[0] + box[2] / 2) * dw
    center_y = (box[1] + box[3] / 2) * dh
    width = box[2] * dw
    height = box[3] * dh
    return center_x, center_y, width, height


# 转换 VisDrone 数据集标注文件为 YOLO 格式
def visdrone2yolo(dir):
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # 创建 labels 文件夹
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_path = (dir / 'images' / f.name).with_suffix('.jpg')
        img_size = Image.open(img_path).size  # 获取图像的宽和高
        lines = []
        with open(f, 'r') as file:  # 读取原始的 VisDrone 标注文件
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' 类别为 0，跳过这些标注
                    continue
                cls = int(row[5]) - 1  # 将类别索引调整为 YOLO 格式（从 0 开始）
                box = tuple(map(int, row[:4]))  # 获取边界框坐标
                yolo_box = convert_box(img_size, box)  # 转换为 YOLO 格式
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in yolo_box)}\n")

        # 将 YOLO 格式的标签写入到对应的 labels 文件夹
        label_path = str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}')
        with open(label_path, 'w') as fl:
            fl.writelines(lines)  # 保存转换后的标注


# 设置数据集路径
dataset_dir = Path(r'D:\ultralytics-main\visdrone')  # 修改为你数据集的路径
# 执行转换，遍历训练集、验证集和测试集
for subdir in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
    visdrone2yolo(dataset_dir / subdir)
