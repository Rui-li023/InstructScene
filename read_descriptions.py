import pickle
import os

def read_pkl_file():
    # 定义文件路径
    file_path = "dataset/InstructScene/InstructScene/threed_front_bedroom/0b4abd30-b157-4ecf-a077-989285598cf2_SecondBedroom-6482/descriptions.pkl"
    # file_path = "dataset/InstructScene/InstructScene/threed_front_bedroom/0b4abd30-b157-4ecf-a077-989285598cf2_SecondBedroom-6482/models_info.pkl"
    # file_path = "dataset/InstructScene/InstructScene/threed_future_model_bedroom.pkl"
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
