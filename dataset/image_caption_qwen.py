import os
import base64
import time
import json
from tqdm import tqdm
import asyncio
import aiofiles
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from io import BytesIO
import PIL.Image as Image

# 配置参数
DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.siliconflow.cn/v1/",
    "temperature": 0.3,
    "concurrent_limit": 20
}

def get_client(api_key=None, base_url=None):
    """初始化OpenAI客户端"""
    api_key = api_key or DEFAULT_CONFIG["api_key"]
    base_url = base_url or DEFAULT_CONFIG["base_url"]
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def encode_image_to_base64(image_path):
    image = Image.open(image_path)
    if image.mode in ["RGBA", "P"]:
        image = image.convert("RGB")
    
    # 计算缩放比例，保持宽高比
    width, height = image.size
    if width > height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
    
    # 使用LANCZOS重采样方法进行缩放
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def load_model_info():
    """Load model information from model_info.json file"""
    json_path = os.path.join(os.path.dirname(__file__), 'InstructScene/3D-FRONT/3D-FUTURE-model/model_info.json')
    async with aiofiles.open(json_path, 'r') as f:
        model_info = json.loads(await f.read())
    # 创建一个以model_id为key的字典,便于查找
    return {item['model_id']: item for item in model_info}

async def get_image_description(image_path, category=None, sem=None, max_retries=3, client=None, temperature=None):
    base64_image = encode_image_to_base64(image_path)
    temperature = temperature or DEFAULT_CONFIG["temperature"]
    
    prompt = "You are looking at a piece of furniture. Please describe it in one short sentence. " + \
             "please combine their information and generate a new short description in one line. " + \
             "Focus on important aspects like color, shape, and material. " + \
             "The new description should be as short and concise as possible, encoded in ASCII. " + \
             "Do not describe the background and counting numbers. " + \
             "Do not describe size like 'small', 'large', etc. " + \
             "Do not include descrptions like 'a 3D model', 'a 3D image', 'a 3D printed', etc. " + \
             "Keep the description concise and objective.\n" + \
             "Here is a emample for your answer:\n" + \
             "- a brown multi-seat sofa with wooden legs\n" + \
             "- a pendant lamp with hanging balls\n" + \
             "- a black and brown floral armchair\n" + \
             "- a conical lamp with a glass and dark metal accents.\n" + \
             "and you must include the category name or a part of category in the description. The sentence must start with a/an \n"
    
    if category:
        prompt += f"The category name information: {category}\n"
        
    for retry in range(max_retries):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=400
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
            if retry < max_retries - 1:
                await asyncio.sleep(2 ** retry)
            else:
                return None

async def process_model(model_id, base_dir, output_dir, model_info_dict, sem, client, temperature):
    output_file = os.path.join(output_dir, f"{model_id}.txt")
    image_path = os.path.join(base_dir, model_id, "image.jpg")
    
    if not os.path.exists(image_path) or os.path.exists(output_file):
        return
        
    model_info = model_info_dict.get(model_id, {})
    category = model_info.get('category', '')
    
    description = await get_image_description(image_path, category, sem, 3, client, temperature)
    
    if description:
        async with aiofiles.open(output_file, "w", encoding='utf-8') as f:
            await f.write(description)
        print(f"{model_id}: {description}")

async def main(api_key=None, base_url=None, temperature=None, concurrent_limit=None):
    base_dir = "InstructScene/3D-FRONT/3D-FUTURE-model"
    output_dir = "/home/ubuntu/Documents/ACDC_project/InstructScene/dataset/InstructScene/3D-FUTURE-qwen"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化客户端
    client = get_client(api_key, base_url)
    concurrent_limit = concurrent_limit or DEFAULT_CONFIG["concurrent_limit"]
    
    # 加载模型信息
    model_info_dict = await load_model_info()
    model_ids = list(model_info_dict.keys())
    
    sem = asyncio.Semaphore(concurrent_limit)  # 限制并发数
    
    tasks = [process_model(model_id, base_dir, output_dir, model_info_dict, sem, client, temperature) 
             for model_id in model_ids]
    
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        try:
            await coro
        except Exception as e:
            print(f"Task failed: {str(e)}")
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    asyncio.run(main())
