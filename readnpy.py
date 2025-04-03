import numpy as np

# 文件路径
file_path = "dataset/InstructScene/InstructScene/threed_front_bedroom/00110bde-f580-40be-b8bb-88715b338a2a_Bedroom-43072/openshape_pointbert_vitg14.npy"

# 读取 .npy 文件
try:
    data = np.load(file_path)
    
    # 打印基本信息
    print(f"数据类型: {type(data)}")
    print(f"数组形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"前5个元素:\n {data[:5]}")
    
    # 如果想查看更多统计信息
    print(f"最小值: {np.min(data)}")
    print(f"最大值: {np.max(data)}")
    print(f"平均值: {np.mean(data)}")

except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确")
except Exception as e:
    print(f"发生错误: {str(e)}")