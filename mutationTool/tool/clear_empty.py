import os
import json

def delete_folder(folder_path):
    """删除指定文件夹"""
    print(folder_path)
    os.system('rm -rf ' + folder_path)



def traverse_folder(root_path):
    """遍历指定目录下的mutantResult.json文件"""
    with open("./failVersion.json", 'r') as f:
        failVersion = json.load(f)
    failNum = 0
    passNum = 0
    failDict = dict()
    for project in os.listdir(root_path):
        # project = "Math"
        for version in os.listdir(root_path + "/" + project):
            # if failVersion.get(project) and version in failVersion[project]:
            #     continue
            status_count = {'0': 0, '1': 0}  # 记录status为0和1的数量
            file_path = os.path.join(root_path, project, version, "muResult.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for i in range(0, len(data)):
                        status_count[str(data[i]['status'])] += 1
                if status_count['0'] > status_count['1'] / 5 or status_count['1'] == 0:
                    # delete_folder(root_path + "/" + project + "/" + version)  # 删除文件所在的文件夹
                    # print(project, version)
                    if failDict.get(project) is None:
                        failDict[project] = list()
                    failDict[project].append(version)
                    failNum += 1
                else:
                    passNum += 1
            except:
                # delete_folder(root_path + "/" + project + "/" + version)  # 删除文件所在的文件夹
                # print(project, version)
                if failDict.get(project) is None:
                    failDict[project] = list()
                failDict[project].append(version)
                failNum += 1
        if failDict.get(project):
            failDict[project] = sorted(failDict[project])
        # break
    print(f"failNum: {failNum}")
    print(f"passNum: {passNum}")
    print(failDict)
    with open("./t_failVersion.json", 'w') as f:
        f.write(json.dumps(failDict, indent=2))
        
if __name__ =="__main__":
    traverse_folder("/home/fanluxi/pmbfl/faultlocalizationResult")