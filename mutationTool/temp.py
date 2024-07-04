import os
import shutil

def copy_files(source_dir, destination_dir):
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 拼接源文件路径
            source_file_path = os.path.join(root, file)
            
            # 判断是否符合要求的文件
            if "susFunction" in root and file.endswith(".json"):
                # 构建目标目录路径
                destination_sub_dir = os.path.relpath(root, source_dir)
                destination_dir_path = os.path.join(destination_dir, destination_sub_dir)
                
                # 如果目标目录不存在，则创建
                if not os.path.exists(destination_dir_path):
                    os.makedirs(destination_dir_path)
                print(source_file_path, destination_dir_path)
                # 拷贝文件到目标目录
                shutil.copy(source_file_path, destination_dir_path)
            # 判断是否符合要求的文件
            if file.endswith("muInfo.json") or file.endswith("faultLocalization.json"):
                # 构建目标目录路径
                destination_sub_dir = os.path.relpath(root, source_dir)
                destination_dir_path = os.path.join(destination_dir, destination_sub_dir)
                
                # 如果目标目录不存在，则创建
                if not os.path.exists(destination_dir_path):
                    os.makedirs(destination_dir_path)
                print(source_file_path, destination_dir_path)
                # 拷贝文件到目标目录
                shutil.copy(source_file_path, destination_dir_path)
                shutil.copy(source_file_path, destination_dir_path)
            # 判断是否符合要求的文件
            if "susStatement" in root and file.endswith(".json"):
                # 构建目标目录路径
                destination_sub_dir = os.path.relpath(root, source_dir)
                destination_dir_path = os.path.join(destination_dir, destination_sub_dir)
                
                # 如果目标目录不存在，则创建
                if not os.path.exists(destination_dir_path):
                    os.makedirs(destination_dir_path)
                print(source_file_path, destination_dir_path)
                # 拷贝文件到目标目录
                shutil.copy(source_file_path, destination_dir_path)

if __name__ == "__main__":
    source_directory = "/home/fanluxi/pmbfl/FOMfaultlocalizationResult/Csv"
    destination_directory = "/home/fanluxi/pmbfl/SOMfaultlocalizationResult/Csv"
    
    copy_files(source_directory, destination_directory)