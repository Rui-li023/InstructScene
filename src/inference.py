import os
import open_clip
import torch
import numpy as np
import pickle
from PIL import Image
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
import trimesh
from typing import Dict, Tuple, Optional

from torch.utils.data import DataLoader
from src.data import get_dataset_raw_and_encoded, filter_function
from src.utils.util import load_config, load_checkpoints
from src.utils.visualize import draw_scene_graph
from src.data.threed_future_dataset import ThreedFutureDataset
from src.data.utils_text import fill_templates, model_desc_from_info
from src.models import model_from_config, ObjectFeatureVQVAE
from src.models.sg2sc_diffusion import Sg2ScDiffusion
from src.models.clip_encoders import CLIPTextEncoder
from src.models.sg_diffusion_vq_objfeat import scatter_trilist_to_matrix
from src.utils import export_scene, blender_render_scene, floor_plan_from_scene
from src.utils.util import *
from src.utils.visualize import *

def load_custom_furniture(furniture_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    从furniture_output目录加载生成的家具资产
    返回: {category: (obj_path, texture_path)}的字典
    """
    furniture_dict = {}
    for category in os.listdir(furniture_dir):
        category_dir = os.path.join(furniture_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # 查找.obj文件和对应的材质文件
        obj_file = None
        texture_file = None
        for file in os.listdir(category_dir):
            if file.endswith('.glb'):
                obj_file = os.path.join(category_dir, file)
            elif file.endswith(('.png', '.jpg', '.jpeg')):
                texture_file = os.path.join(category_dir, file)
        
        if obj_file and texture_file:
            furniture_dict[category] = (obj_file, texture_file)
            
    return furniture_dict


class SceneGenerator:
    def __init__(self, 
                 config_path,
                 fvqvae_path,
                 sg2sc_path,
                 output_dir="output",
                 custom_furniture_dir="furniture_output",
                 device="cuda:0"):
        """
        初始化场景生成器
        Args:
            config_path: 配置文件路径
            fvqvae_path: fVQ-VAE模型checkpoint路径
            sg2sc_path: Sg2Sc模型checkpoint路径
            output_dir: 输出目录
            custom_furniture_dir: 自定义家具资产目录
            device: 使用的设备
        """
        self.device = torch.device(device)
        self.config = load_config(config_path)
            
        # 加载3D模型数据集
        self.objects_dataset = ThreedFutureDataset.from_pickled_dataset(
            self.config["data"]["path_to_pickled_3d_futute_models"]
        )
        
        # 加载文本编码器
        self.clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
        
        # 加载主生成模型
        self.model = model_from_config(
            self.config["model"],
            21,
            10,
            use_objfeat="objfeat" in self.config["model"]["name"]
        ).to(self.device)
        print(self.config["model"])
        # 加载EMA状态
        ema_states = EMAModel(self.model.parameters())
        load_checkpoints(self.model, os.path.dirname(sg2sc_path), ema_states, device=self.device)
        ema_states.copy_to(self.model.parameters())
        self.model.eval()

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载自定义家具资产
        self.custom_furniture = load_custom_furniture(custom_furniture_dir)
        
        self.raw_dataset, self.dataset = get_dataset_raw_and_encoded(
            self.config["data"],
            filter_fn=filter_function(
                self.config["data"],
                split=self.config["validation"].get("splits", ["test"])
            ),
            path_to_bounds=None,
            augmentations=None,
            split=self.config["validation"].get("splits", ["test"])
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=self.dataset.collate_fn,
            shuffle=False
        )
        
    def get_custom_furniture_mesh(self, category: str, bbox_params: ndarray) -> Optional[trimesh.Trimesh]:
        """获取自定义家具的trimesh对象"""
        if category not in self.custom_furniture:
            return None
            
        obj_path, _ = self.custom_furniture[category]
        tr_mesh = trimesh.load_mesh(obj_path, force="mesh")

        # 获取变换参数，调换-5,-6和-3,-4的顺序
        translation = bbox_params[-7:-4].copy()   # [x,y,z]
        sizes = bbox_params[-4:-1].copy()  # [x,y,z]
        # translation[[1, 2]] = translation[[2, 1]]
        # sizes[[1, 2]] = sizes[[2, 1]]
        theta = bbox_params[-1]

        # 围绕Y轴的旋转矩阵
        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])

        # 目标尺寸
        target_size = sizes  # 使用调整后的尺寸
        current_size = tr_mesh.bounds[1] - tr_mesh.bounds[0]

        print(tr_mesh.bounds[0])
        print(tr_mesh.bounds[1])
        
        # 计算缩放因子并使用最小值进行等比例缩放
        scale_factors = target_size / current_size
        min_scale = np.min(scale_factors)
        scale = np.array([min_scale, min_scale, min_scale])

        # 按顺序应用变换:
        # 1. 缩放
        tr_mesh.vertices *= scale_factors

        # 2. 居中
        center = (tr_mesh.bounds[0] + tr_mesh.bounds[1])
        
        # print(tr_mesh.bounds[0])
        # print(tr_mesh.bounds[1])
        
        # tr_mesh.vertices -= center

        # 3. 应用旋转（现在是绕Y轴）
        tr_mesh.vertices = tr_mesh.vertices.dot(R)

        # 4. 应用平移
        translation[1] *= 0.5
        tr_mesh.vertices += translation

        return tr_mesh

    def generate(self, 
                text_description,
                objs,
                edges,
                mask,
                gpt_output,
                floor_plan=None,
                cfg_scale=1.0,
                sg2sc_cfg_scale=1.0,
                save_vis=True,
                resolution=1024):
        """
        根据文本描述生成3D场景
        Args:
            text_description: 场景的文本描述
            floor_plan: 地板平面图路径(可选)
            cfg_scale: 分类器自由引导的缩放系数
            sg2sc_cfg_scale: sg2sc模型的缩放系数
            save_vis: 是否保存可视化结果
            resolution: 渲染分辨率
        Returns:
            generated_scene: 包含场景信息的字典
        """
        with torch.no_grad():
            # 1. 处理文本输入
            print(text_description)
            text = self.tokenizer(text_description)
            text_features = self.clip.encode_text(text)
            
            # 填充text_features到11的长度
            batch_size, feat_dim = text_features.shape
            if batch_size < 11:
                # 创建随机填充
                padding = torch.rand(11 - batch_size, feat_dim, device=text_features.device) * 4 - 2
                text_features = torch.cat([text_features, padding], dim=0)
            
            # 添加批次维度
            text_features = text_features.unsqueeze(0)  # shape: [1, 11, 1280]
            
            # 2. 生成场景图
            objs = objs.to(self.device)
            obj_masks = mask.to(self.device)
            edges = edges.to(self.device)
            text_features = text_features.to(self.device)
            room_masks = floor_plan.to(self.device)
            
            # for batch in self.dataloader:
            #     for k, v in batch.items():
            #         if not isinstance(v, list):
            #             batch[k] = v.to(self.device)
            #     objs = batch["objs"].to(self.device)
            #     edges = batch["edges"].to(self.device)
            #     obj_masks = batch["obj_masks"].to(self.device)
            #     room_masks = batch["room_masks"].to(self.device)  # (B, 256, 256)
            #     text_features = batch['clip_features'].to(self.device)

            #     boxes_pred = self.model.generate_samples(
            #         objs, edges, text_features, obj_masks, room_masks=room_masks, cfg_scale=cfg_scale, num_timesteps=1000
            #     )
            #     print(objs)
            #     print(edges)
            #     print(obj_masks)
            #     print(boxes_pred)
            #     print(batch["boxes"])
            
            print(objs)
            print(edges)
            print(obj_masks)
            boxes_pred = self.model.generate_samples(
                objs, edges, text_features, obj_masks, room_masks=room_masks, cfg_scale=cfg_scale, num_timesteps=1000
            )
            
            print(boxes_pred.shape)
            print(boxes_pred)
        
        objs, edges = objs.cpu(), edges.cpu()
        boxes_pred = boxes_pred.cpu()

        bbox_params = {
            "class_labels": F.one_hot(objs, num_classes=self.dataset.n_object_types+1).float(),
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:]
        }
            
        boxes = self.dataset.post_process(bbox_params)
        boxes = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).numpy()
        
        print(boxes)
        
        # 从生成的boxes创建场景
        trimesh_meshes = []
        bbox_meshes = []
        obj_classes = []
        obj_sizes = []
        obj_ids = []

        for i in range(boxes.shape[1]):
            if mask[0, i] == 0:  # 跳过空物体
                continue
                
            obj_class = gpt_output["objects"][i]["category"]
            obj_classes.append(obj_class)
            obj_sizes.append(boxes[0, i, -4:-1])
            obj_ids.append(f"generated_{i}")
            
            # 尝试加载自定义家具,如果没有则使用默认数据集
            custom_mesh = self.get_custom_furniture_mesh(model_desc_from_info(obj_class), boxes[0, i])
            if custom_mesh is not None:
                trimesh_meshes.append(custom_mesh)

        # 添加地板
        if floor_plan is not None:
            tr_floor, _ = floor_plan_from_scene(
                self.raw_dataset[0],
                self.config["data"]["path_to_floor_plan_textures"],
                without_room_mask=True,
                rectangle_floor=True
            )
            trimesh_meshes.append(tr_floor)
            
        if save_vis:
            # 创建输出目录
            export_dir = os.path.join(self.output_dir, "scene")
            tmp_dir = os.path.join(export_dir, "tmp")
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(tmp_dir, exist_ok=True)
            
            # 导出场景
            export_scene(tmp_dir, trimesh_meshes, bbox_meshes)
            
            # 使用Blender渲染
            blender_render_scene(
                tmp_dir,
                export_dir,
                resolution_x=resolution,
                resolution_y=resolution
            )
            
            # 清理临时文件
            # os.system(f"rm -rf {tmp_dir}")
            
        return {
            "meshes": trimesh_meshes,
            "bbox_meshes": bbox_meshes,
            "object_classes": obj_classes,
            "object_sizes": obj_sizes,
            "object_ids": obj_ids
        }
