import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
# from torch.utils.tensorboard import SummaryWriter

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).eval()
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']
        
    def forward(self, x, return_logits=False):
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[f'style_{name}'] = x
            if name in self.content_layers:
                features[f'content_{name}'] = x
        
        if return_logits:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return features, x
        return features


def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)


class StyleTransfer:
    def __init__(self, content_img, style_img, device='cuda', target_class_name='cinema', original_class_name='barn'):
        self.device = device
        self.content_img = content_img.to(device)
        self.style_img = style_img.to(device)
        self.model = VGGFeatures().to(device)
        self.generated = self.content_img.clone().to(device).requires_grad_(True)

        # 加载 ImageNet 类别标签
        with open('utils/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        # 获取目标类别的索引
        if target_class_name in classes:
            self.target_class = classes.index(target_class_name)
            print(f"类别 '{target_class_name}' 的索引是: {self.target_class}")
        else:
            raise ValueError(f"类别 '{target_class_name}' 不在 ImageNet 类别列表中。")

    def compute_content_loss(self, gen_features, content_features):
        content_loss = 0
        for key in content_features:
            if key.startswith('content_'):
                content_loss += torch.mean((gen_features[key] - content_features[key])**2)
        return content_loss
    
    def compute_style_loss(self, gen_features, style_features):
        style_loss = 0
        for key in style_features:
            if key.startswith('style_'):
                gen_gram = gram_matrix(gen_features[key])
                style_gram = gram_matrix(style_features[key])
                style_loss += torch.mean((gen_gram - style_gram)**2)
        return style_loss

    
    def compute_adversarial_loss(self, logits):
        # 对抗损失：目标类别的交叉熵损失
        target = torch.tensor([self.target_class] * logits.size(0), device=logits.device)
        adv_loss = torch.nn.functional.cross_entropy(logits, target)
        return adv_loss
    
    # def compute_original_class_loss(self, logits):
    #     # 原始类别的交叉熵损失
    #     original_target = torch.tensor([self.original_class] * logits.size(0), device=logits.device)
    #     orig_loss = torch.nn.functional.cross_entropy(logits, original_target)
    #     return orig_loss
    
    def train(self, num_steps=300, style_weight=1e2, content_weight=5.0, adv_weight=5e3, log_interval=5, learning_rate=0.02, optimizer_type='adam'):
        content_features = self.model(self.content_img)
        style_features = self.model(self.style_img)
        
        # 根据选择的优化器类型初始化优化器
        if optimizer_type == 'adam':
            optimizer = optim.Adam([self.generated], lr=learning_rate)
        elif optimizer_type == 'lbfgs':
            optimizer = optim.LBFGS([self.generated])
        else:
            raise ValueError("不支持的优化器类型。请选择 'adam' 或 'lbfgs'。")
        
        # # TensorBoard日志记录
        # writer = SummaryWriter(log_dir="tensorboard_logs")

        best_loss = float('inf')
        best_img = None
        best_losses = None

        pbar = tqdm(range(num_steps), desc="训练进度")
        
        def closure():
            optimizer.zero_grad()
            gen_features, logits = self.model(self.generated, return_logits=True)
            
            content_loss = self.compute_content_loss(gen_features, content_features)
            style_loss = self.compute_style_loss(gen_features, style_features)
            adv_loss = self.compute_adversarial_loss(logits)
            
            total_loss = (content_weight * content_loss + 
                          style_weight * style_loss + 
                          adv_weight * adv_loss)
            
            nonlocal best_loss, best_img, best_losses
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_img = self.generated.clone().detach()
                best_losses = {
                    'step': step,
                    'content_loss': content_loss.item(),
                    'style_loss': style_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'total_loss': total_loss.item()
                }
            
            total_loss.backward(retain_graph=True)
            
            # # 每10步记录一次到 TensorBoard
            # if step % log_interval == 0:
            #     writer.add_scalar('Loss/Content_Loss', content_loss.item(), step)
            #     writer.add_scalar('Loss/Style_Loss', style_loss.item(), step)
            #     writer.add_scalar('Loss/Adversarial_Loss', adv_loss.item(), step)
            #     writer.add_scalar('Loss/Total_Loss', total_loss.item(), step)
            
            pbar.set_postfix({
                'content_loss': f'{content_loss.item():.4f}',
                'style_loss': f'{style_loss.item():.4f}',
                'adv_loss': f'{adv_loss.item():.4f}',
                'total_loss': f'{total_loss.item():.4f}'
            })
            
            return total_loss

        for step in pbar:
                optimizer.step(closure)
        
        # writer.close()  # 关闭 TensorBoard 日志

        return best_img if best_img is not None else self.generated.detach(), best_losses
