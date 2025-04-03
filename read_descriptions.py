import pickle
import os

def read_pkl_file():
    # 定义文件路径
    file_path = "dataset/InstructScene/InstructScene/threed_front_bedroom/00110bde-f580-40be-b8bb-88715b338a2a_Bedroom-43072/models_info.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    try:
        # 读取PKL文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # 打印数据内容
        print("PKL文件内容：")
        print(data)
        
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

if __name__ == "__main__":
    read_pkl_file()
