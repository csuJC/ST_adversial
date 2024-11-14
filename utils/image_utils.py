import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path, size=224):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    print("预处理完成")
    return image

def get_imagenet_label(pred):
    """获取ImageNet标签和概率"""
    # 加载ImageNet类别标签
    with open('utils/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    probabilities = torch.nn.functional.softmax(pred, dim=1)
    _, indices = torch.max(pred, 1)
    return classes[indices.item()], probabilities[0, indices.item()].item()

def save_image(tensor, path):
    """保存张量为图像"""
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor * std + mean
    tensor.clamp_(0, 1)
    
    # 转换为PIL图像并保存
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(path)

def get_target_class_probability(output, target_class_name):
    """
    获取目标类别的预测概率
    
    Args:
        output: 模型输出的预测结果
        target_class_name: 目标类别名称（如 'cinema'）
    
    Returns:
        float: 目标类别的预测概率
    """
    # 加载ImageNet类别名称映射
    with open('utils/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 找到目标类别的索引
    target_idx = classes.index(target_class_name)
    # print(target_idx)
    # 计算softmax概率
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 返回目标类别的概率
    return probabilities[target_idx].item()