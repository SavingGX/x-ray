# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

classes = {
    0: 'USBFlashDisk', 1: 'battery', 2: 'knife', 3: 'lighter',
    4: 'plasticBottleWithaNozzle', 5: 'pressure', 6: 'scissors', 7: 'seal'
}

def main():
    # 加载模型
    model = YOLO('./pre_train_models/yolov8n.pt')  # 加载预训练模型

    # 训练模型
    results = model.train(data='./data/x-ray.yaml', batch=16 ,epochs=100 ,workers=4 ,device=0)

    # 测试图像路径
    img = "./data/tests/001617.jpg"
    name = img.split('/')[-1].split('.')[0]

    # 模型预测
    model = YOLO('runs/detect/train2/weights/best.pt')
    num = len(os.listdir('runs/detect')) - 2  # 计算预测结果目录编号
    model.predict(source=img, save=True, save_txt=True, conf=0.4)

    # 构造标签文件路径
    file_path = f'runs/detect/predict{num}/labels/{name}.txt'

    # 检查标签文件是否存在
    if not os.path.exists(file_path):
        print(f"未检测到目标，标签文件未生成: {file_path}")
    else:
        with open(file_path, 'r') as f:
            a = int(f.read().split(' ')[0])
            print(classes[a])

if __name__ == '__main__':
    main()