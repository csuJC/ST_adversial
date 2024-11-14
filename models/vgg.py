import torch
import torchvision.models as models

def load_vgg19():
    """加载预训练的VGG19模型"""
    # model = models.vgg19()
    # # 从本地加载预训练权重
    # model.load_state_dict(torch.load('models/pretrained/vgg19-dcbb9e9d.pth'))
    model = models.vgg19(pretrained=True)  # 使用预训练权重
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model 