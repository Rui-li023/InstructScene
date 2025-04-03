import openai
import json
import os
from pathlib import Path

openai.api_key = "sk-uUtSQKCMVkTY5QDMzKR232unNfPF6UiCs4jhDsnk0RzlZN9B"
openai.base_url = "https://api.claudeshop.top/v1/"

def parse_items(layout_result):
    """解析布局结果，返回物品列表"""
    items = []
    for line in layout_result.split('\n'):
        if '.' in line:
            item = line.strip().split('.')[1].strip()
            # 存储标准化的物品名称
            items.append(item)
    return items

class QACache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, prompt):
        """获取缓存文件路径"""
        import hashlib
        # 使用prompt的hash作为文件名
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self.cache_dir / f"{prompt_hash}.json"
    
    def get(self, prompt):
        """从缓存中获取回答"""
        cache_path = self._get_cache_path(prompt)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save(self, prompt, response):
        """保存回答到缓存"""
        cache_path = self._get_cache_path(prompt)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

def generate_pre_formatted_pairs(layout_result):
    # 初始化结果列表
    pre_formatted_pairs = []
    # 计数器，用于编号
    counter = 1
    
    # 遍历 layout_result 中的所有唯一物体对
    for i in range(len(layout_result)):
        for j in range(i + 1, len(layout_result)):
            # 格式化每一对物体
            pair = f"{counter}. '{layout_result[i]}' [relationship] '{layout_result[j]}'"
            pre_formatted_pairs.append(pair)
            counter += 1
    
    # 将列表转换为字符串，用换行符连接
    return "\n".join(pre_formatted_pairs)

# 创建cache实例
qa_cache = QACache()

def get_gpt_response(prompt, model="gpt-4o", temperature=0.7):
    """获取GPT响应，如果有缓存则使用缓存"""
    # 先检查缓存
    cached_response = qa_cache.get(prompt)
    if cached_response:
        return type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': cached_response
                })
            })]
        })

    # 如果没有缓存，调用API
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful interior design assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    
    # 保存到缓存
    qa_cache.save(prompt, response.choices[0].message.content)
    return response

furniture_list = [
    "desk", "nightstand", "king-size bed", "single bed", "kids bed",
    "ceiling lamp", "pendant lamp", "bookcase/jewelry armoire", "tv stand",
    "wardrobe", "lounge chair/cafe chair/office chair", "dining chair",
    "classic chinese chair", "armchair", "dressing table", "dressing chair",
    "corner/side table", "dining table", "round end table",
    "drawer chest/corner cabinet", "sideboard/side cabinet/console table",
    "children cabinet", "shelf", "footstool/sofastool/bed end stool/stool",
    "coffee table", "loveseat sofa", "three-seat/multi-seat sofa",
    "l-shaped sofa", "lazy sofa", "chaise longue sofa"
]

# 允许的关系词汇列表
relationship_list = [
    "above", "left of", "in front of",
    "closely left of", "closely in front of",
    "below", "right of", "behind",
    "closely right of", "closely behind"
]

# 获取用户指令
user_instruction = "bedroom for two adults"

# 第一次提示词：生成房间布局
prompt1 = f"""
You are an interior design assistant. Based on the user's request: '{user_instruction}', 
generate a room layout with a list of items and their quantities. 
Choose appropriate furniture and decorations only from the following list: {', '.join(furniture_list)}. 

### Guidelines:
1. Ensure the layout is functional.
2. Don't describe the spatial relationships between the items in this prompt.
3. Your output items do not need to be capitalized and must be the same as the list provided.
4. If you think the room needs more items of one type, you can add them, the number represents id, e.g., 'Item A(1)', 'Item A(2)'.
5. You can add 11 objects to the bedroom at most, but usually 4-8 objects are suitable

### Output format:
Return the result in a clear list format, for example:
===BEGIN===

1. Item 1
2. Item 2(1)
3. Item 2(2)

===END===


"""

# 调用 OpenAI API 获取第一次回复
response1 = get_gpt_response(prompt1)

# 提取第一次回复结果
layout_result = response1.choices[0].message.content
print(f"Generated room layout for '{user_instruction}':")
print(layout_result)

items = parse_items(layout_result)
pre_formatted_pairs = generate_pre_formatted_pairs(items)

print(pre_formatted_pairs)

# 第二次提示词：基于第一次回复，要求描述物体间的相互关系
prompt2 = f"""

You are an interior design assistant. Your task is to determine the positional relationship between pairs of items in a room based on a user-requested layout: '{user_instruction}'.

### Room items are provided:
{layout_result}

### Instructions:
You will be given pre-formatted pairs of items in the output section below.
For each pair, fill in the [relationship] placeholder with one of the following terms: {', '.join(relationship_list)}.
Do not modify the item names, numbers, or the structure of the output; only provide the relationship term.
If multiple instances of an item exist (e.g., two chairs), they are already distinguished as 'item(1)', 'item(2)', etc., in the pair.
Each pair is unique, and you should provide the relationship based on a logical spatial interpretation of the layout.

### Output format:
The pairs are listed below. Replace [relationship] with the appropriate term from the provided list.

===BEGIN===

{pre_formatted_pairs}

===END===

"""




# 调用 OpenAI API 获取第二次回复
response2 = get_gpt_response(prompt2)

# 提取并打印第二次回复结果
relationship_result = response2.choices[0].message.content
print(f"\nrelationships for '{user_instruction}' layout:")
print(relationship_result)

# 关系映射字典
relationship_mapping = {
    "above": 1,
    "left of": 2,
    "in front of": 3,
    "closely left of": 4,
    "closely in front of": 5,
    "below": 6,
    "right of": 7,
    "behind": 8,
    "closely right of": 9,
    "closely behind": 10
}

# 第三次提示词：生成物体的属性描述
prompt3 = f"""
You are a furniture selection designer and your goal is to give a description of each piece of furniture for the user's requirements.
Based on the room layout for '{user_instruction}', generate a simple description for each item that includes its material, color, or other key attributes.

### Room items:
{layout_result}

### Instructions:
1. For each item, provide a brief description (e.g., "a metal coffee table with a flat surface", "a black and gold cabinet with drawers")
2. For the same type, if you think they are the same, you must provide the same descriptions. But you should use different numbers to distinguish them, e.g., 'Item A(1)', 'Item A(2)'.
3. Your descriptions should be as easy as possible.
4. Your output items do not need to be capitalized and must be the same as the list provided.
5. output with ' and ".

### Output format:
===BEGIN===

1. 'Item A': "description"
2. 'Item B(1)': "description"

===END===
"""

# 第四次提示词：生成用于3D生成的描述
prompt4 = f"""
For each item in the room layout, generate a detailed description suitable for 2D image generation.

### Room items:
{layout_result}

### Instructions:
1. Focus on physical characteristics, style, and structural details
2. Include specific dimensions, materials, and design elements
3. Make descriptions detailed enough for 2D modeling
4. Use one sentence for each item
5. Your output items do not need to be capitalized and must be the same as the list provided.

### Output format:
===BEGIN===

1. 'Item A': "detailed description for 2D generation"
2. 'Item B(1)': "detailed description for 2D generation"

===END===
"""

# 调用API获取属性描述
response3 = get_gpt_response(prompt3)
print(response3.choices[0].message.content)

# 调用API获取2D生成描述
# response4 = get_gpt_response(prompt4)
# print(response4.choices[0].message.content)

def normalize_item_name(item):
    """标准化物品名称，移除编号"""
    return item.split('(')[0].strip()

def parse_relationships(relationship_result):
    """解析关系描述，返回关系列表"""
    relationships = []
    for line in relationship_result.split('\n'):
        if "'" in line:
            rel = line.split("'")[1::2]  # 提取引号中的内容
            if len(rel) >= 2:
                obj1, obj2 = rel[0], rel[1]
                for rel_type in relationship_mapping.keys():
                    if rel_type in line:
                        relationships.append((obj1, obj2, rel_type))
                        break
    return relationships

def reverse_rel(rel):
    """反转关系"""
    reverse_mapping = {
        "above": "below",
        "below": "above",
        "left of": "right of",
        "right of": "left of",
        "in front of": "behind",
        "behind": "in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of"
    }
    return reverse_mapping.get(rel, rel)

def create_relationship_matrix(items, relationships):
    """创建完整的关系矩阵（包括上三角和下三角）"""
    n = len(items)
    matrix = [[0] * n for _ in range(n)]
    
    # 创建物品到索引的映射
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    
    # 填充关系矩阵
    for obj1, obj2, rel in relationships:
        i = item_to_idx[obj1]
        j = item_to_idx[obj2]
        # 上三角矩阵
        if i < j:
            matrix[i][j] = relationship_mapping[rel]
            # 填充对应的下三角矩阵
            opposite_rel = reverse_rel(rel)
            matrix[j][i] = relationship_mapping[opposite_rel]
        # 下三角矩阵
        else:
            matrix[i][j] = relationship_mapping[rel]
            # 填充对应的上三角矩阵
            opposite_rel = reverse_rel(rel)
            matrix[j][i] = relationship_mapping[opposite_rel]
    
    return matrix

def parse_descriptions(description_result):
    """解析描述结果"""
    descriptions = {}
    for line in description_result.split('\n'):
        if "'" in line and ':' in line:
            item = line.split("'")[1]
            desc = line.split('"')[1]
            # 使用标准化的物品名称作为键
            descriptions[normalize_item_name(item)] = desc
    return descriptions

# 处理结果

print(items)
relationships = parse_relationships(relationship_result)
semantic_desc = parse_descriptions(response3.choices[0].message.content)
# model_desc = parse_descriptions(response4.choices[0].message.content)

# 创建关系矩阵
relationship_matrix = create_relationship_matrix(items, relationships)

# 构建最终输出
final_result = {
    "objects": [
        {
            "name": item,
            "category": normalize_item_name(item),
            "semantic_description": semantic_desc.get(item, ""),
            # "model_description": model_desc.get(item, "")
        }
        for item in items
    ],
    "relationship_matrix": relationship_matrix
}

print(relationship_matrix)
print("\nFinal Result:")
print(final_result)