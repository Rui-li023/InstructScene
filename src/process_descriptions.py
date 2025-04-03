import os
import torch
import open_clip
import numpy as np
from tqdm import tqdm

def process_descriptions():
    # 设置输入和输出目录
    input_dir = "/home/ubuntu/Documents/ACDC_project/InstructScene/dataset/InstructScene/InstructScene/3D-FUTURE-chatgpt"
    output_dir = "/home/ubuntu/Documents/ACDC_project/InstructScene/dataset/InstructScene/InstructScene/3D-FUTURE-chatgpt-clip-feature"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    
    # 获取所有描述文件
    description_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # 处理每个描述文件
    for filename in tqdm(description_files, desc="Processing descriptions"):
        # 读取描述文本
        with open(os.path.join(input_dir, filename), 'r') as f:
            description = f.read().strip()
        
        # 生成CLIP特征
        with torch.no_grad():
            text = tokenizer(description)
            text = text.to(device)
            text_features = model.encode_text(text)
            text_features = text_features.cpu().numpy()
        
        # 保存特征
        output_path = os.path.join(output_dir, filename.replace('.txt', '.npy'))
        np.save(output_path, text_features[0])

if __name__ == "__main__":
    process_descriptions()
