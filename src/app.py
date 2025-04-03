import os
import random
import torch
import trimesh
from pathlib import Path
import uuid
import shutil
import json
import gc
from typing import List, Dict

from data.utils_text import model_desc_from_info
from hy3dgen.shapegen import (
    FaceReducer,
    FloaterRemover, 
    DegenerateFaceRemover,
    Hunyuan3DDiTFlowMatchingPipeline
)
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.rembg import BackgroundRemover

from diffusers import StableDiffusion3Pipeline

class Text2Model:
    def __init__(self):
        # 初始化保存目录
        self.save_dir = "output"
        self.temp_dir = "temp_output"  # 用于保存中间结果
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 后处理工具（这些工具占用内存较少，可以常驻）
        self.floater_remover = FloaterRemover()
        self.face_reducer = FaceReducer()
        self.degenerate_remover = DegenerateFaceRemover()
        self.rmbg_worker = BackgroundRemover()

    def _load_t2i_model(self):
        # model = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.half)
        model.enable_model_cpu_offload()
        return model

    def _load_shape_model(self):
        model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            subfolder='hunyuan3d-dit-v2-0-turbo',
            use_safetensors=True
        )
        model.enable_flashvdm()
        return model

    def _load_texture_model(self):
        return Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

    def _clear_gpu_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def generate_image(self, prompt: str, save_path: str):
        model = self._load_t2i_model()
        image = model(prompt)
        image = self.rmbg_worker(image)
        image.save(save_path)
        del model
        self._clear_gpu_memory()
        return image

    def generate_shape(self, image, steps: int, guidance_scale: float, seed: int, save_path: str):
        model = self._load_shape_model()
        generator = torch.Generator().manual_seed(seed)
        outputs = model(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type='mesh'
        )
        mesh = export_to_trimesh(outputs)[0]
        mesh = self.floater_remover(mesh)
        mesh = self.degenerate_remover(mesh)
        mesh = self.face_reducer(mesh)
        mesh.export(save_path)
        del model
        self._clear_gpu_memory()
        return mesh

    def generate_texture(self, mesh, image, save_path: str):
        model = self._load_texture_model()
        textured_mesh = model(mesh, image)
        textured_mesh.export(save_path)
        del model
        self._clear_gpu_memory()
        return textured_mesh

    def generate_furniture_set(self, furniture_list: List[Dict], output_dir: str = "furniture_output") -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 第1阶段：批量生成所有图像
        print("\n===== 第1阶段：生成所有家具的图像 =====")
        model_t2i = self._load_t2i_model()
        image_paths = {}
        
        for item in furniture_list:
            category = item['category']
            # description = item['model_description']
            description = item['semantic_description']
            
            category_dir = os.path.join(output_dir, model_desc_from_info(category))
            os.makedirs(category_dir, exist_ok=True)
            
            try:
                enhanced_prompt = f"a {category}, {description}, Minimalist elegance, Natural textures, Soft, muted tones, Functional simplicity, Cozy warmth, front view, white background, realistic, minimalistic design, high detail, centered, studio lighting, no people, clean edges"
                
                temp_image_path = os.path.join(self.temp_dir, f"{model_desc_from_info(category)}_temp.png")
                final_image_path = os.path.join(category_dir, f"{model_desc_from_info(category)}.png")
                
                image = model_t2i(enhanced_prompt).images[0]
                image = self.rmbg_worker(image)
                image.save(temp_image_path)
                shutil.copy2(temp_image_path, final_image_path)
                
                image_paths[category] = {
                    'temp_path': temp_image_path,
                    'final_path': final_image_path,
                    'image': image
                }
                print(f"成功生成 {category} 的图像")
                
            except Exception as e:
                print(f"生成 {category} 的图像失败: {str(e)}")
                results[category] = {'status': 'failed', 'error': str(e)}
        
        del model_t2i
        self._clear_gpu_memory()
        
        # 第2阶段：批量生成所有3D形状
        print("\n===== 第2阶段：生成所有3D形状 =====")
        model_shape = self._load_shape_model()
        mesh_paths = {}
        
        for category, paths in image_paths.items():
            if category not in results:  # 跳过之前失败的项目
                try:
                    temp_mesh_path = os.path.join(self.temp_dir, f"{model_desc_from_info(category)}_temp.obj")
                    seed = random.randint(0, int(1e7))
                    
                    mesh = self.generate_shape(
                        paths['image'], 
                        steps=30, 
                        guidance_scale=5.0,
                        seed=seed,
                        save_path=temp_mesh_path
                    )
                    
                    mesh_paths[category] = {
                        'temp_path': temp_mesh_path,
                        'mesh': mesh
                    }
                    print(f"成功生成 {category} 的3D形状")
                    
                except Exception as e:
                    print(f"生成 {category} 的3D形状失败: {str(e)}")
                    results[category] = {'status': 'failed', 'error': str(e)}
        
        del model_shape
        self._clear_gpu_memory()
        
        # 第3阶段：批量生成所有纹理
        print("\n===== 第3阶段：生成所有纹理 =====")
        model_texture = self._load_texture_model()
        
        for category, mesh_data in mesh_paths.items():
            if category not in results:  # 跳过之前失败的项目
                try:
                    category_dir = os.path.join(output_dir, model_desc_from_info(category))
                    final_model_path = os.path.join(category_dir, f"{model_desc_from_info(category)}.glb")
                    
                    textured_mesh = model_texture(
                        mesh_data['mesh'],
                        image_paths[category]['image']
                    )
                    textured_mesh.export(final_model_path)
                    
                    results[category] = {
                        'model_path': final_model_path,
                        'image_path': image_paths[category]['final_path'],
                        'status': 'success'
                    }
                    print(f"成功生成 {category} 的纹理")
                    
                except Exception as e:
                    print(f"生成 {category} 的纹理失败: {str(e)}")
                    results[category] = {'status': 'failed', 'error': str(e)}
        
        del model_texture
        self._clear_gpu_memory()
        
        # 清理所有临时文件
        for category in image_paths:
            if os.path.exists(image_paths[category]['temp_path']):
                os.remove(image_paths[category]['temp_path'])
        
        for category in mesh_paths:
            if os.path.exists(mesh_paths[category]['temp_path']):
                os.remove(mesh_paths[category]['temp_path'])
        
        # 保存生成结果信息
        result_file = os.path.join(output_dir, 'generation_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def main():
    generator = Text2Model()
    
    # 从文件读取或直接定义家具列表
    furniture_list = [{'name': 'king-size bed', 'category': 'king-size bed', 'semantic_description': ''}, {'name': 'nightstand(1)', 'category': 'nightstand', 'semantic_description': ''}, {'name': 'nightstand(2)', 'category': 'nightstand', 'semantic_description': ''}, {'name': 'wardrobe', 'category': 'wardrobe', 'semantic_description': ''}, {'name': 'drawer chest/corner cabinet', 'category': 'drawer chest/corner cabinet', 'semantic_description': ''}, {'name': 'ceiling lamp', 'category': 'ceiling lamp', 'semantic_description': ''}, {'name': 'armchair', 'category': 'armchair', 'semantic_description': ''}]
    # 生成所有家具的3D模型
    results = generator.generate_furniture_set(furniture_list)
    
    # 打印生成结果
    print("\n生成结果摘要:")
    for category, result in results.items():
        status = result['status']
        if status == 'success':
            print(f"{category}: 成功 - 模型保存在 {result['model_path']}")
        else:
            print(f"{category}: 失败 - {result.get('error', '未知错误')}")

if __name__ == "__main__":
    main()
