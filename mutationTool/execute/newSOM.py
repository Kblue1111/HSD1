import os
import json
import sys

with open('./config.json', 'r') as configFile:
    configData = json.load(configFile)
SOMfaultlocalizationResultPath = configData['SOMfaultlocalizationResultPath']


#按照每个程序平均每个版本的变异体数量约减
def sub(project, version, somInfoList):
    try:
        # 定义变异体信息存储位置
        subSOMInfoPath = os.path.join(
            SOMfaultlocalizationResultPath, project, version, "newsubSOMInfo.json"
        )
        
        if os.path.exists(subSOMInfoPath):
            os.remove(subSOMInfoPath)
            print(f"Removed existing file at {subSOMInfoPath}")
            
        # 检查文件是否已经存在，如果不存在则创建并保存子集
        if not os.path.exists(subSOMInfoPath):
            # 直接从somInfoList中取前xx个变异体作为子集
            subSOMInfoList = somInfoList[:525]

            # 保存子集到JSON文件
            with open(subSOMInfoPath, "w") as f:
                json.dump(subSOMInfoList, f, indent=2)
            
            print(f"Saved {len(subSOMInfoList)} mutations to {subSOMInfoPath}")
        else:
            # 如果文件已存在，直接加载现有数据
            with open(subSOMInfoPath, "r") as f:
                subSOMInfoList = json.load(f)
            print(f"Loaded {len(subSOMInfoList)} mutations from {subSOMInfoPath}")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        return None

    return subSOMInfoList

if __name__ == '__main__':
    project = "Chart"
    version = "26b"
    with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "SOMInfo.json"), 'r') as f:
        somInfoList = json.load(f)
    sub_SOM = sub(project, version, somInfoList)
    