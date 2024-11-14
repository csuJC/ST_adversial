import torch
from models.vgg import load_vgg19
from utils.image_utils import load_image, get_imagenet_label
import os
import argparse

def test_vgg19(image_name):
    # 构建完整的图片路径
    image_path = os.path.join("data/content_images/last", image_name)
    if not image_name.endswith('.jpg'):
        image_path += '.jpg'
    
    # 加载模型
    model = load_vgg19()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查图片路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    
    # 加载并预处理图片
    image = load_image(image_path)
    image = image.to(device)
    
    # 进行预测
    with torch.no_grad():
        output = model(image)
    
    # 获取预测结果
    predicted_label = get_imagenet_label(output)
    
    print(f"预测类别: {predicted_label}")

def pre():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='VGG19 图像分类测试')
    parser.add_argument('--image', type=str, default="styled_image.jpg",
                      help='图片名称 (默认: 1)')
    
    args = parser.parse_args()
    test_vgg19(args.image)

if __name__ == "__main__":
    pre()
