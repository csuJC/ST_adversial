import torch
from utils.image_utils import load_image, save_image, get_imagenet_label, get_target_class_probability
from models.style_transfer import StyleTransfer
import argparse
import os
from tqdm import tqdm
from models.vgg import load_vgg19

def test_style_transfer(args):
    if args.loop:
        # 处理last文件夹下的所有图片
        content_dir = os.path.join("data/content_images/last")
        output_dir = os.path.join("results/last")
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有jpg图片
        content_images = [f for f in os.listdir(content_dir) if f.endswith('.jpg')]
        
        for img_name in content_images:
            # 构建路径
            content_path = os.path.join(content_dir, img_name)
            style_path = os.path.join("data/style_images", args.style_image)
            if not style_path.endswith('.jpg'):
                style_path += '.jpg'
            
            # 设置输出路径
            output_name = f"{os.path.splitext(img_name)[0]}_{os.path.splitext(args.style_image)[0]}.jpg"
            output_path = os.path.join(output_dir, output_name)
            
            print(f"处理图片: {img_name}")
            process_single_image(content_path, style_path, output_path, args)
    else:
        # 处理单张图片
        content_path = os.path.join("data/content_images", args.content_image)
        style_path = os.path.join("data/style_images", args.style_image)
        output_dir = os.path.join("results")
        # 设置输出路径
        output_name = f"{os.path.splitext(args.content_image)[0]}_{os.path.splitext(args.style_image)[0]}.jpg"
        output_path = os.path.join(output_dir, output_name)
        if not content_path.endswith('.jpg'):
            content_path += '.jpg'
        if not style_path.endswith('.jpg'):
            style_path += '.jpg'
        
        process_single_image(content_path, style_path, output_path, args)

def process_single_image(content_path, style_path, output_path, args):
    # 检查文件是否存在
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"找不到内容图片: {content_path}")
    if not os.path.exists(style_path):
        raise FileNotFoundError(f"找不到风格图片: {style_path}")
    
    # 加载图片
    print("正在加载图片...")
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建风格迁移模型
    style_transfer = StyleTransfer(content_img, style_img, device, target_class_name=args.target_class_name)
    
    # 进行风格迁移
    print("开始风格迁移...")
    generated_img, losses = style_transfer.train(
        num_steps=args.num_steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        adv_weight=args.adv_weight,
        optimizer_type = args.optimizer_type
    )
    
    # 保存生成的对抗图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(generated_img, output_path)
    print(f"结果已保存到: {output_path}")
    
    # 加载 VGG19 模型并预测生成图像
    model = load_vgg19().to(device)
    generated_img = generated_img.to(device)  # 将生成图像加载到设备上
    with torch.no_grad():
        output = model(generated_img)
    
    # 获取预测结果和概率
    predicted_label, probability = get_imagenet_label(output)
    target_probability = get_target_class_probability(output, args.target_class_name)
    # 准备记录结果
    result_file = "results/style_transfer_results.txt"
    os.makedirs("results", exist_ok=True)
    
    result_str = (f"图片: {os.path.basename(output_path)}, "
                  f"轮数: {losses['step']}, "
                  f"内容损失: {losses['content_loss']:.4f}, "
                  f"风格损失: {losses['style_loss']:.4f}, "
                  f"对抗损失: {losses['adv_loss']:.4f}, "
                  f"总损失: {losses['total_loss']:.4f}, "
                  f"预测类别: {predicted_label}, "
                  f"预测概率: {probability:.4f}")
    
    # 写入结果
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(result_str + '\n')
    
    print(f"预测结果已写入: {result_file}")

def main():
    parser = argparse.ArgumentParser(description='风格迁移与对抗样本生成')
    
    # 添加参数
    parser.add_argument('--content_image', type=str, default='dog',
                      help='内容图片名称 (默认: dog)')
    parser.add_argument('--style_image', type=str, default='fire',
                      help='风格图片名称 (默认: fire)')
    parser.add_argument('--num_steps', type=int, default=2000,
                      help='训练迭代次数 (默认: 2000)')
    parser.add_argument('--content_weight', type=float, default=3,
                      help='内容损失权重 (默认: 3)')
    parser.add_argument('--style_weight', type=float, default=2e4,
                      help='风格损失权重 (默认: 2e4)')
    parser.add_argument('--adv_weight', type=float, default=3,
                      help='对抗损失权重 (默认: 10)')
    parser.add_argument('--loop', action='store_true',
                      help='是否处理last文件夹下的所有图片')
    parser.add_argument('--target_class_name', type=str,default='cinema',
                      help='目标类别名称 (例如: cinema)')
    parser.add_argument('--optimizer_type',type=str,default='adam')
    
    # 先解析参数
    args = parser.parse_args()
    test_style_transfer(args)

if __name__ == "__main__":
    main()