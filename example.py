from PIL import Image
from src.inference import SceneGenerator
import torch
import numpy as np
from src.data.base import THREED_FRONT_BEDROOM_FURNITURE

def process_gpt_output(gpt_output):
    # 创建类别到索引的映射
    #  = {"nightstand": 0, "double_bed": 1, "wardrobe": 2, "pendant_lamp": 3, "ceiling_lamp": 4, "tv_stand": 5, "chair": 6, "single_bed": 7, "dressing_table": 8, "cabinet": 9, "table": 10, "desk": 11, "stool": 12, "shelf": 13, "kids_bed": 14, "bookshelf": 15, "children_cabinet": 16, "dressing_chair": 17, "armchair": 18, "sofa": 19, "coffee_table": 20}
    category_to_idx = {
        "armchair": 0,
        "bookshelf": 1,
        "cabinet": 2,
        "ceiling_lamp": 3,
        "chair": 4,
        "children_cabinet": 5,
        "coffee_table": 6,
        "desk": 7,
        "double_bed": 8,
        "dressing_chair": 9,
        "dressing_table": 10,
        "kids_bed": 11,
        "nightstand": 12,
        "pendant_lamp": 13,
        "shelf": 14,
        "single_bed": 15,
        "sofa": 16,
        "stool": 17,
        "table": 18,
        "tv_stand": 19,
        "wardrobe": 20
        }
    objs = torch.full((1, 11), 21)  # 创建1x11的tensor，默认值为21(空)
    mask = torch.zeros(1, 11, dtype=torch.int)  # 创建1x11的mask，默认值为0
    semantic_descriptions = []
    
    for i, obj in enumerate(gpt_output['objects']):
        if i >= 11:  # 最多处理11个物体
            break
        # 获取标准化的类别
        category = THREED_FRONT_BEDROOM_FURNITURE.get(obj['category'], None)
        if category is not None:
            idx = category_to_idx.get(category, 21)
            objs[0, i] = idx
            mask[0, i] = 1
        semantic_descriptions.append(obj['semantic_description'])
    
    # 处理关系矩阵
    # 创建11x11的边矩阵，填充10表示无关系
    relationship_matrix = gpt_output['relationship_matrix']

    edges = torch.full((1, 11, 11), 10, dtype=torch.long)  
    
    # 将原始关系矩阵填充到11x11矩阵的左上角
    num_objs = min(len(gpt_output['objects']), 11)
    for i in range(num_objs):
        for j in range(num_objs):
            if relationship_matrix[i][j] != 0:
                edges[0, i, j] = relationship_matrix[i][j] - 1  # 将1-10映射到0-9
    
    return objs, edges, mask, semantic_descriptions

device="cuda:0"
# 初始化生成器
generator = SceneGenerator(
    config_path="configs/bedroom_sgfl2lt_diffusion_objfeat.yaml",
    fvqvae_path="out/threedfront_objfeat_vqvae/checkpoints/epoch_01999.pth",
    sg2sc_path="out/2025-03-31_23:09_bedroom_sgfl2lt_diffusion_objfeat.yaml/checkpoints/epoch_00089.pth",
    output_dir="output",
    device=device
)

# gpt_output = {'objects': [{'name': 'king-size bed', 'category': 'king-size bed', 'semantic_description': 'a wooden bed frame with a tufted upholstered headboard in light gray fabric'}, {'name': 'nightstand(1)', 'category': 'nightstand', 'semantic_description': 'a nightstand'}, {'name': 'nightstand(2)', 'category': 'nightstand', 'semantic_description': 'a nightstand'}, {'name': 'wardrobe', 'category': 'wardrobe', 'semantic_description': 'a tall wooden wardrobe with a white matte finish and sliding doors'}, {'name': 'drawer chest/corner cabinet', 'category': 'drawer chest/corner cabinet', 'semantic_description': 'a mid-height chest of drawers in a natural oak finish with sleek metal handles'}, {'name': 'ceiling lamp', 'category': 'ceiling lamp', 'semantic_description': 'a modern pendant ceiling lamp with a frosted glass shade and brushed nickel accents'}, {'name': 'armchair', 'category': 'armchair', 'semantic_description': 'a cushioned armchair with soft beige upholstery and wooden legs'}], 'relationship_matrix': [[0, 2, 7, 3, 3, 6, 3], [7, 0, 2, 3, 2, 6, 2], [2, 7, 0, 3, 7, 6, 7], [8, 8, 8, 0, 2, 6, 8], [8, 7, 2, 7, 0, 6, 7], [1, 1, 1, 1, 1, 0, 1], [8, 7, 2, 3, 2, 6, 0]]}
gpt_output = {'objects': [{'name': 'king-size bed', 'category': 'king-size bed', 'semantic_description': 'a single bed with sheet and pillows'}, 
                          {'name': 'nightstand(1)', 'category': 'nightstand', 'semantic_description': 'a nightstand with a plant on top'}, 
                          {'name': 'nightstand(2)', 'category': 'nightstand', 'semantic_description': 'a nightstand with a plant on top'}, 
                          {'name': 'wardrobe', 'category': 'wardrobe', 'semantic_description': 'a tall wooden wardrobe'}, 
                          {'name': 'drawer chest/corner cabinet', 'category': 'drawer chest/corner cabinet', 'semantic_description': 'a white cabinet with wooden drawers'}, 
                          {'name': 'ceiling lamp', 'category': 'ceiling lamp', 'semantic_description': 'a black pendant lamp with a shade'}], 
                        #   {'name': 'armchair', 'category': 'armchair', 'semantic_description': 'a cushioned armchair'}], 
              'relationship_matrix': [[0, 7, 2, 0, 2, 6], 
                                      [2, 0, 0, 0, 0, 6], 
                                      [7, 0, 0, 2, 0, 6], 
                                      [0, 0, 7, 0, 0, 6], 
                                      [7, 0, 0, 0, 0, 6], 
                                      [1, 1, 1, 1, 1, 0]]}
# 处理GPT输出
objs, edges, mask, semantic_descriptions = process_gpt_output(gpt_output)

from PIL import Image
import torchvision.transforms as transforms

# 打开图像并转换为灰度格式
image = Image.open("floor_plan.png").convert("L")

# 定义二值化阈值
threshold = 128

# 将图像二值化
binary_image = image.point(lambda x: 0 if x < threshold else 255, '1')

# 转换为张量
transform = transforms.ToTensor()
image_tensor = transform(binary_image)

# 确保只保留一个通道
image_tensor = image_tensor[0].unsqueeze(0)  # 保留第一个通道并添加通道维度

# print(objs)
# print(image_tensor.shape)  # 现在输出形状应该是 (1, H, W)，即一个通道、高度和宽度
# print(semantic_descriptions)

scene = generator.generate(
    text_description=semantic_descriptions,
    objs=objs,
    edges=edges,
    mask=mask,
    floor_plan=image_tensor,
    cfg_scale=1.0,
    sg2sc_cfg_scale=1.0,
    save_vis=True,
    resolution=2048,
    gpt_output=gpt_output
)

# 场景结果包含以下内容:
# scene["meshes"] - 3D网格列表
# scene["bbox_meshes"] - 边界框网格列表
# scene["object_classes"] - 物体类别列表
# scene["object_sizes"] - 物体尺寸列表
# scene["object_ids"] - 物体ID列表
